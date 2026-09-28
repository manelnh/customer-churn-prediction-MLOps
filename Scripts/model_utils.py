import os
import pickle
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
try:
    import mlflow
except ModuleNotFoundError:
    mlflow = None
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

DATA_FILE = Path(__file__).resolve().parents[1] / 'telco_churn_cleaned.csv'
ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_BUNDLE_PATH = ROOT_DIR / 'models' / 'churn_production_bundle.pkl'
LEGACY_MODEL_BUNDLE_PATH = ROOT_DIR / 'churn_production.pkl'
ROOT_MODEL_BUNDLE_PATH = ROOT_DIR / 'churn_production_bundle.pkl'
DEFAULT_DROP_COLUMNS = ['Support_Tickets', 'App_Logins']
PRODUCTION_BASELINE_PARAMS = {'C': 100.0, 'solver': 'lbfgs', 'penalty': 'l2', 'class_weight': 'balanced'}
PRODUCTION_BASELINE_METRICS = {'accuracy': 0.867, 'f1': 0.785, 'roc_auc': 0.951}
MIN_LIVE_LABEL_SAMPLE = 20
DEFAULT_DECISION_THRESHOLD = 0.5
MODEL_REGISTRY_NAME = 'lr'
MLFLOW_BUNDLE_ARTIFACT_PATH = 'production_bundle/churn_production_bundle.pkl'
PRODUCTION_MODEL_ALIAS = 'champion'
CANDIDATE_MODEL_ALIAS = 'candidate'


def get_latest_model_version_by_alias_or_stage(
    client,
    model_name: str,
    alias: str = PRODUCTION_MODEL_ALIAS,
    stage: str | None = 'Production',
):
    """Prefer model aliases, with a stage-based fallback for older registry state."""
    try:
        return client.get_model_version_by_alias(model_name, alias)
    except Exception:
        if stage is None:
            raise

    versions = client.search_model_versions(f"name = '{model_name}'")
    stage_matches = [
        version for version in versions
        if getattr(version, 'current_stage', None) == stage
    ]
    if not stage_matches:
        raise FileNotFoundError(f'No model version found for "{model_name}" using alias "{alias}" or stage "{stage}".')
    return max(stage_matches, key=lambda version: int(version.version))


def validate_model_bundle(bundle: dict):
    model = bundle['model']
    dv = bundle['dv']
    scaler = bundle['scaler']

    dv_feature_count = len(dv.get_feature_names_out())
    scaler_feature_count = getattr(scaler, 'n_features_in_', None)
    model_feature_count = getattr(model, 'n_features_in_', None)

    if dv_feature_count == scaler_feature_count == model_feature_count:
        return model, dv, scaler

    raise ValueError(
        'Incompatible bundle: '
        f'dv={dv_feature_count}, scaler={scaler_feature_count}, model={model_feature_count}'
    )


def load_latest_production_bundle_from_mlflow(
    model_name: str = MODEL_REGISTRY_NAME,
    artifact_path: str = MLFLOW_BUNDLE_ARTIFACT_PATH,
) -> dict:
    if mlflow is None:
        raise ModuleNotFoundError('mlflow is not installed')

    mlflow_tracking_uri = get_mlflow_tracking_uri()
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = mlflow.MlflowClient()
    latest_version = get_latest_model_version_by_alias_or_stage(client, model_name, alias=PRODUCTION_MODEL_ALIAS, stage='Production')
    local_artifact_path = mlflow.artifacts.download_artifacts(
        run_id=latest_version.run_id,
        artifact_path=artifact_path,
    )
    bundle = load_bundle(local_artifact_path)
    metadata = bundle.setdefault('metadata', {})
    metadata.setdefault('mlflow_model_name', model_name)
    metadata.setdefault('mlflow_model_version', latest_version.version)
    metadata.setdefault('mlflow_run_id', latest_version.run_id)
    creation_timestamp = getattr(latest_version, 'creation_timestamp', None)
    if creation_timestamp:
        deployed_at = datetime.fromtimestamp(creation_timestamp / 1000, tz=timezone.utc).isoformat()
        metadata.setdefault('deployed_at', deployed_at)
    return bundle

DEFAULT_DECISION_THRESHOLD = 0.51


def classify_from_probability(probability: float, threshold: float = DEFAULT_DECISION_THRESHOLD) -> str:
    return 'Churn' if probability >= threshold else 'No churn'


def calculate_model_selection_score(metrics: dict) -> float:
    f1 = metrics.get('f1', 0.0)
    roc_auc = metrics.get('roc_auc', 0.0)
    accuracy = metrics.get('accuracy', 0.0)
    log_loss_value = metrics.get('log_loss', 0.0)
    return f1 + 0.1 * roc_auc + 0.05 * accuracy - 0.05 * log_loss_value


def filter_predictions_for_active_model(predictions, bundle_metadata):
    if predictions is None or predictions.empty or not bundle_metadata:
        return predictions

    filtered = predictions.copy()
    run_id = bundle_metadata.get('mlflow_run_id') or bundle_metadata.get('run_id')
    variant_name = bundle_metadata.get('variant_name')

    if run_id and 'mlflow_run_id' in filtered.columns:
        matches = filtered[filtered['mlflow_run_id'] == run_id]
        if not matches.empty:
            return matches.reset_index(drop=True)

    if variant_name and 'model_version' in filtered.columns:
        matches = filtered[filtered['model_version'] == variant_name]
        if not matches.empty:
            return matches.reset_index(drop=True)

    return filtered


def get_active_bundle_metadata() -> dict:
    for path in (MODEL_BUNDLE_PATH, LEGACY_MODEL_BUNDLE_PATH):
        if path.exists():
            try:
                return load_bundle(path).get('metadata', {}) or {}
            except Exception:
                continue
    return {}


def get_baseline_metrics_from_metadata(bundle_metadata: dict | None) -> dict:
    bundle_metadata = bundle_metadata or {}
    baseline = bundle_metadata.get('baseline_metrics') or bundle_metadata.get('production_metrics')
    if isinstance(baseline, dict) and baseline:
        return {
            'accuracy': baseline.get('accuracy', PRODUCTION_BASELINE_METRICS['accuracy']),
            'f1': baseline.get('f1', PRODUCTION_BASELINE_METRICS['f1']),
            'roc_auc': baseline.get('roc_auc', PRODUCTION_BASELINE_METRICS['roc_auc']),
        }
    return get_production_baseline_metrics()

def load_model_bundle():
    """Load the production model bundle (model, dict vectorizer, scaler)."""
    ...

def load_model_bundle():
    """Load the production model bundle (model, dict vectorizer, scaler)."""
    bundle = load_active_model_bundle()
    return validate_model_bundle(bundle)


def load_active_model_bundle() -> dict:
    """Load the active production bundle with metadata."""
    bundle_loaders = []
    if mlflow is not None:
        bundle_loaders.append(
            (
                'MLflow Production registry',
                lambda: load_latest_production_bundle_from_mlflow(),
            )
        )
    bundle_loaders.extend(
        [
            ('legacy local bundle', lambda: load_bundle(LEGACY_MODEL_BUNDLE_PATH)),
            ('local production bundle', lambda: load_bundle(MODEL_BUNDLE_PATH)),
            ('root production bundle', lambda: load_bundle(ROOT_MODEL_BUNDLE_PATH)),
        ]
    )
    last_error = None

    for source_name, bundle_loader in bundle_loaders:
        try:
            bundle = bundle_loader()
            validate_model_bundle(bundle)
            return bundle
        except Exception as exc:
            last_error = RuntimeError(f'{source_name} could not be loaded: {exc}')

    if last_error is not None:
        raise last_error

    raise FileNotFoundError(
        'No model bundle found in MLflow Production registry or local bundle paths: '
        f'{LEGACY_MODEL_BUNDLE_PATH}, {MODEL_BUNDLE_PATH}'
    )


def load_bundle_metadata(path: str | Path) -> dict:
    bundle = load_bundle(path)
    return bundle.get('metadata', {})


def get_active_bundle_metadata() -> dict:
    try:
        bundle = load_active_model_bundle()
        return bundle.get('metadata', {})
    except Exception:
        return {}


def get_baseline_metrics_from_metadata(metadata: dict | None = None) -> dict:
    metadata = metadata or {}
    baseline = metadata.get('test_metrics') or metadata.get('val_metrics') or {}
    if not baseline:
        return get_production_baseline_metrics()
    merged = get_production_baseline_metrics()
    merged.update({key: value for key, value in baseline.items() if value is not None})
    return merged


def filter_predictions_for_active_model(logs: pd.DataFrame, metadata: dict | None = None) -> pd.DataFrame:
    if logs.empty:
        return logs.copy()

    metadata = metadata or {}
    filtered = logs.copy()
    mlflow_run_id = metadata.get('mlflow_run_id') or metadata.get('run_id') or metadata.get('source_candidate_run_id')
    variant_name = metadata.get('variant_name')
    model_registry_version = metadata.get('mlflow_model_version')
    deployed_at = metadata.get('deployed_at')

    if deployed_at and 'created_at' in filtered.columns:
        deployed_at_ts = pd.to_datetime(deployed_at, errors='coerce', utc=True)
        created_at_ts = pd.to_datetime(filtered['created_at'], errors='coerce', utc=True)
        if pd.notna(deployed_at_ts):
            deployed_window = filtered[created_at_ts >= deployed_at_ts].copy()
            if not deployed_window.empty:
                filtered = deployed_window

    if mlflow_run_id and 'mlflow_run_id' in filtered.columns:
        run_filtered = filtered[filtered['mlflow_run_id'] == mlflow_run_id].copy()
        if not run_filtered.empty:
            return run_filtered

    if variant_name and 'model_version' in filtered.columns:
        variant_filtered = filtered[filtered['model_version'] == variant_name].copy()
        if not variant_filtered.empty:
            return variant_filtered

    if model_registry_version and 'model_version' in filtered.columns:
        registry_filtered = filtered[filtered['model_version'] == str(model_registry_version)].copy()
        if not registry_filtered.empty:
            return registry_filtered

    return filtered


def get_production_baseline_params():
    """Get production baseline parameters."""
    return PRODUCTION_BASELINE_PARAMS.copy()


def get_production_baseline_metrics():
    """Get production baseline metrics."""
    return PRODUCTION_BASELINE_METRICS.copy()


def load_data(csv_path: str | Path = None) -> pd.DataFrame:
    path = Path(csv_path) if csv_path else DATA_FILE
    return pd.read_csv(path)


def split_data(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.25, random_state: int = 42):
    df_full_train, df_test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['Churn'] if 'Churn' in df.columns else None,
    )
    df_train, df_val = train_test_split(
        df_full_train,
        test_size=val_size,
        random_state=random_state,
        stratify=df_full_train['Churn'] if 'Churn' in df_full_train.columns else None,
    )
    return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)


def encode_target(series: pd.Series) -> np.ndarray:
    normalized = series.fillna('No').astype(str)
    return np.where(normalized.isin(['Yes', 'Churn', '1']), 1, 0)


def apply_feature_engineering(df_original: pd.DataFrame) -> pd.DataFrame:
    df = df_original.copy()

    service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'StreamingTV', 'StreamingMovies']
    for col in service_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)

    if 'SeniorCitizen' in df.columns:
        df['SeniorCitizen'] = df['SeniorCitizen'].map({'Yes': 1, 'No': 0, 1: 1, 0: 0}).fillna(0).astype(int)

    if 'tenure' in df.columns and 'MonthlyCharges' in df.columns:
        available_service_cols = [col for col in service_cols if col in df.columns]
        if available_service_cols:
            df['TotalServices'] = df[available_service_cols].sum(axis=1)
        df['Total_Revenue'] = df['tenure'] * df['MonthlyCharges']
        df['Monthly_per_Tenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)
        df['Is_First_Year'] = (df['tenure'] <= 12).astype(int)

    if 'Data_Usage_GB' in df.columns and 'tenure' in df.columns:
        df['Usage_per_Month'] = df['Data_Usage_GB'] / (df['tenure'] + 1)

    return df


def drop_unused_columns(df: pd.DataFrame, to_drop: list[str] = None) -> pd.DataFrame:
    to_drop = to_drop or DEFAULT_DROP_COLUMNS
    return df.drop(columns=to_drop, errors='ignore')


def prepare_features(df: pd.DataFrame, dv: DictVectorizer = None, fit_dv: bool = False):
    feature_dicts = df.to_dict(orient='records')
    if dv is None:
        dv = DictVectorizer(sparse=False)
        X = dv.fit_transform(feature_dicts)
    elif fit_dv:
        X = dv.fit_transform(feature_dicts)
    else:
        X = dv.transform(feature_dicts)
    return X, dv


def scale_features(X: np.ndarray, scaler: StandardScaler = None, fit_scaler: bool = False):
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    elif fit_scaler:
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    return X_scaled, scaler


def prepare_modeling_datasets(
    df: pd.DataFrame,
    drop_columns: list[str] | None = None,
    test_size: float = 0.2,
    val_size: float = 0.25,
    random_state: int = 42,
) -> dict:
    df_train, df_val, df_test = split_data(
        df,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
    )

    y_train = encode_target(df_train['Churn'])
    y_val = encode_target(df_val['Churn'])
    y_test = encode_target(df_test['Churn'])

    df_train_features = drop_unused_columns(apply_feature_engineering(df_train.drop(columns=['Churn'])), drop_columns)
    df_val_features = drop_unused_columns(apply_feature_engineering(df_val.drop(columns=['Churn'])), drop_columns)
    df_test_features = drop_unused_columns(apply_feature_engineering(df_test.drop(columns=['Churn'])), drop_columns)

    X_train, dv = prepare_features(df_train_features, fit_dv=True)
    X_val, _ = prepare_features(df_val_features, dv=dv)
    X_test, _ = prepare_features(df_test_features, dv=dv)

    X_train_scaled, scaler = scale_features(X_train, fit_scaler=True)
    X_val_scaled, _ = scale_features(X_val, scaler=scaler)
    X_test_scaled, _ = scale_features(X_test, scaler=scaler)

    return {
        'dv': dv,
        'scaler': scaler,
        'train': {'X': X_train_scaled, 'y': y_train, 'features': df_train_features},
        'val': {'X': X_val_scaled, 'y': y_val, 'features': df_val_features},
        'test': {'X': X_test_scaled, 'y': y_test, 'features': df_test_features},
    }


def train_logistic_regression_variant(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float,
    solver: str,
    penalty: str = 'l2',
    max_iter: int = 1000,
    class_weight: str | None = 'balanced',
    l1_ratio: float | None = None,
    random_state: int = 42,
):
    model = LogisticRegression(
        C=C,
        solver=solver,
        penalty=penalty,
        max_iter=max_iter,
        class_weight=class_weight,
        l1_ratio=l1_ratio,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def build_precision_recall_dataframe(model, X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    probabilities = model.predict_proba(X)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y, probabilities)
    pr_auc = auc(recall, precision)
    threshold_values = np.append(thresholds, 1.0)
    return pd.DataFrame(
        {
            'precision': precision,
            'recall': recall,
            'threshold': threshold_values,
            'pr_auc': pr_auc,
        }
    )


def run_logistic_regression_experiment(
    datasets: dict,
    C: float,
    solver: str,
    penalty: str = 'l2',
    max_iter: int = 1000,
    class_weight: str | None = 'balanced',
    l1_ratio: float | None = None,
    random_state: int = 42,
) -> dict:
    model = train_logistic_regression_variant(
        X_train=datasets['train']['X'],
        y_train=datasets['train']['y'],
        C=C,
        solver=solver,
        penalty=penalty,
        max_iter=max_iter,
        class_weight=class_weight,
        l1_ratio=l1_ratio,
        random_state=random_state,
    )
    val_probabilities = model.predict_proba(datasets['val']['X'])[:, 1]
    best_threshold, _ = determine_best_threshold(datasets['val']['y'], val_probabilities)
    model.decision_threshold_ = best_threshold

    metrics = {
        'train': evaluate_model(model, datasets['train']['X'], datasets['train']['y'], decision_threshold=best_threshold),
        'val': evaluate_model(model, datasets['val']['X'], datasets['val']['y'], decision_threshold=best_threshold),
        'test': evaluate_model(model, datasets['test']['X'], datasets['test']['y'], decision_threshold=best_threshold),
    }

    return {
        'model': model,
        'dv': datasets['dv'],
        'scaler': datasets['scaler'],
        'metrics': metrics,
        'precision_recall': build_precision_recall_dataframe(model, datasets['test']['X'], datasets['test']['y']),
        'params': {
            'model_family': 'logistic_regression',
            'C': C,
            'solver': solver,
            'penalty': penalty,
            'max_iter': max_iter,
            'class_weight': class_weight,
            'l1_ratio': l1_ratio,
            'decision_threshold': best_threshold,
        },
    }


def train_classifier(X_train: np.ndarray, y_train: np.ndarray, param_grid: dict | None = None):
    param_grid = param_grid or {
        'C': [0.01, 0.1, 1, 10, 100],
        'max_iter': [100, 1000, 2000],
        'solver': ['liblinear', 'lbfgs'],
        'class_weight': ['balanced'],
    }
    lr = LogisticRegression(random_state=42)
    grid_search = GridSearchCV(
        lr,
        param_grid,
        cv=5,
        scoring='f1',
        verbose=1,
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)
    return grid_search


def determine_best_threshold(
    y_true: np.ndarray,
    y_probabilities: np.ndarray,
    min_threshold: float = 0.20,
    max_threshold: float = 0.80,
    step: float = 0.01,
) -> tuple[float, float]:
    y_true = np.asarray(y_true)
    y_probabilities = np.asarray(y_probabilities)
    best_threshold = DEFAULT_DECISION_THRESHOLD
    best_f1 = -1.0
    threshold = min_threshold

    while threshold <= max_threshold + 1e-9:
        y_pred = (y_probabilities >= threshold).astype(int)
        current_f1 = f1_score(y_true, y_pred, zero_division=0)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = round(threshold, 2)
        threshold += step

    return best_threshold, best_f1


def calculate_model_selection_score(metrics: dict) -> float:
    """Weighted validation score used to choose the best production candidate."""
    return (
        0.60 * float(metrics.get('f1', 0.0))
        + 0.30 * float(metrics.get('roc_auc', 0.0))
        + 0.10 * float(metrics.get('accuracy', 0.0))
        - 0.05 * float(metrics.get('log_loss', 0.0))
    )


def classify_from_probability(probability: float, threshold: float = DEFAULT_DECISION_THRESHOLD) -> str:
    return 'Churn' if probability >= threshold else 'No churn'


def evaluate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    decision_threshold: float = DEFAULT_DECISION_THRESHOLD,
) -> dict:
    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_pred_proba >= decision_threshold).astype(int)
    return {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1': f1_score(y, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y, y_pred_proba),
        'log_loss': log_loss(y, y_pred_proba),
        'decision_threshold': decision_threshold,
    }


def save_bundle(
    path: str | Path,
    model,
    dv: DictVectorizer,
    scaler: StandardScaler,
    metadata: dict | None = None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as file:
        pickle.dump({'model': model, 'dv': dv, 'scaler': scaler, 'metadata': metadata or {}}, file)


def log_bundle_artifact_to_mlflow(
    model,
    dv: DictVectorizer,
    scaler: StandardScaler,
    metadata: dict | None = None,
    artifact_dir: str = 'production_bundle',
    filename: str = 'churn_production_bundle.pkl',
):
    if mlflow is None:
        raise ModuleNotFoundError('mlflow is not installed')

    with tempfile.TemporaryDirectory() as temp_dir:
        bundle_path = Path(temp_dir) / filename
        save_bundle(bundle_path, model, dv, scaler, metadata=metadata)
        mlflow.log_artifact(str(bundle_path), artifact_path=artifact_dir)


def load_bundle(path: str | Path):
    path = Path(path)
    with open(path, 'rb') as file:
        return pickle.load(file)


def prepare_customer_features(customer_data: dict, dv: DictVectorizer, scaler: StandardScaler):
    expected_feature_count = getattr(scaler, 'n_features_in_', None)
    actual_feature_count = len(dv.get_feature_names_out())

    if expected_feature_count is not None and actual_feature_count != expected_feature_count:
        _, repaired_dv, repaired_scaler = load_model_bundle()
        dv = repaired_dv
        scaler = repaired_scaler
        expected_feature_count = getattr(scaler, 'n_features_in_', None)
        actual_feature_count = len(dv.get_feature_names_out())

    if expected_feature_count is not None and actual_feature_count != expected_feature_count:
        raise ValueError(
            f'Incompatible prediction bundle: DictVectorizer has {actual_feature_count} features '
            f'but StandardScaler expects {expected_feature_count}.'
        )

    df = pd.DataFrame([customer_data])
    df_engineered = apply_feature_engineering(df)
    df_final = drop_unused_columns(df_engineered)
    X, _ = prepare_features(df_final, dv=dv, fit_dv=False)
    X_scaled, _ = scale_features(X, scaler=scaler, fit_scaler=False)
    return X_scaled, df_final


def explain_top_drivers(X_scaled: np.ndarray, model, dv: DictVectorizer, top_k: int = 3) -> pd.DataFrame:
    feature_names = dv.get_feature_names_out()
    weights = model.coef_[0]
    impact_scores = weights * X_scaled.ravel()
    impact_df = pd.DataFrame({'feature': feature_names, 'impact': impact_scores})
    return impact_df.sort_values(by='impact', ascending=False).head(top_k)


def get_risk_label(probability: float) -> tuple[str, str]:
    if probability >= 0.70:
        return 'HIGH RISK', 'High Risk'
    if probability >= 0.51:
        return 'MEDIUM RISK', 'Medium Risk'
    return 'LOW RISK', 'Low Risk'


def format_feature_name(name: str) -> str:
    return name.replace('=', ': ').replace('_', ' ').title()


def serialize_top_drivers(driver_df: pd.DataFrame) -> list[dict]:
    return [
        {'feature': row['feature'], 'impact': float(row['impact'])}
        for _, row in driver_df.iterrows()
    ]


def prepare_monitoring_dataframe(logs: pd.DataFrame) -> pd.DataFrame:
    if logs.empty:
        return logs.copy()

    df = logs.copy()
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    df['actual_label_at'] = pd.to_datetime(df.get('actual_label_at'), errors='coerce')
    df['predicted_probability'] = pd.to_numeric(df['predicted_probability'], errors='coerce')
    return df.dropna(subset=['created_at', 'predicted_probability']).sort_values('created_at').reset_index(drop=True)


def calculate_live_metrics(logs: pd.DataFrame) -> dict:
    labeled = logs.dropna(subset=['actual_label']).copy()
    if labeled.empty:
        return {'coverage': 0.0, 'sample_size': 0}

    y_true = encode_target(labeled['actual_label'])
    y_pred = encode_target(labeled['predicted_label'])
    y_scores = labeled['predicted_probability'].to_numpy()
    metrics = {
        'coverage': len(labeled) / max(len(logs), 1),
        'sample_size': int(len(labeled)),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    if len(np.unique(y_true)) > 1:
        metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
    else:
        metrics['roc_auc'] = np.nan
    return metrics


def build_probability_drift_frame(logs: pd.DataFrame, resample_frequency: str = 'D') -> pd.DataFrame:
    if logs.empty:
        return pd.DataFrame(columns=['created_at', 'avg_probability', 'high_risk_share'])

    df = logs.copy().set_index('created_at')
    drift = df.resample(resample_frequency).agg(
        avg_probability=('predicted_probability', 'mean'),
        high_risk_share=('predicted_risk', lambda values: (values == 'HIGH RISK').mean()),
        volume=('predicted_probability', 'count'),
    )
    return drift.reset_index().dropna(subset=['avg_probability'])


def detect_monitoring_alerts(
    logs: pd.DataFrame,
    baseline_metrics: dict | None = None,
    recent_window: int = 30,
    min_feedback_coverage: float = 0.30,
    min_live_label_sample: int = MIN_LIVE_LABEL_SAMPLE,
) -> list[dict]:
    baseline_metrics = baseline_metrics or PRODUCTION_BASELINE_METRICS
    alerts: list[dict] = []

    if logs.empty:
        return [
            {
                'type': 'low_prediction_volume',
                'severity': 'medium',
                'message': 'No production predictions are available yet.',
                'current_value': 0.0,
                'baseline_value': None,
                'threshold': 10.0,
            }
        ]

    total_predictions = len(logs)
    if total_predictions < 10:
        alerts.append(
            {
                'type': 'low_prediction_volume',
                'severity': 'medium',
                'message': 'Monitoring sample size is still too small for stable conclusions.',
                'current_value': float(total_predictions),
                'baseline_value': None,
                'threshold': 10.0,
            }
        )

    recent_slice = logs.tail(min(recent_window, total_predictions))
    historical_slice = logs.iloc[:-len(recent_slice)] if total_predictions > len(recent_slice) else pd.DataFrame()

    if not historical_slice.empty:
        current_high_risk = (recent_slice['predicted_risk'] == 'HIGH RISK').mean()
        baseline_high_risk = (historical_slice['predicted_risk'] == 'HIGH RISK').mean()
        if baseline_high_risk > 0:
            increase = (current_high_risk - baseline_high_risk) / baseline_high_risk
            if increase > 0.15:
                alerts.append(
                    {
                        'type': 'high_risk_increase',
                        'severity': 'medium',
                        'message': f'High-risk predictions increased by {increase:.1%}.',
                        'current_value': float(current_high_risk),
                        'baseline_value': float(baseline_high_risk),
                        'threshold': 0.15,
                    }
                )

        current_probability = recent_slice['predicted_probability'].mean()
        baseline_probability = historical_slice['predicted_probability'].mean()
        probability_drift = abs(current_probability - baseline_probability)
        if probability_drift > 0.08:
            alerts.append(
                {
                    'type': 'probability_drift',
                    'severity': 'medium',
                    'message': f'Average prediction probability drift reached {probability_drift:.1%}.',
                    'current_value': float(current_probability),
                    'baseline_value': float(baseline_probability),
                    'threshold': 0.08,
                }
            )

    live_metrics = calculate_live_metrics(logs)
    if live_metrics.get('sample_size', 0) > 0:
        if live_metrics['coverage'] < min_feedback_coverage:
            alerts.append(
                {
                    'type': 'ground_truth_gap',
                    'severity': 'medium',
                    'message': 'Too few predictions have confirmed outcomes for reliable live evaluation.',
                    'current_value': float(live_metrics['coverage']),
                    'baseline_value': 1.0,
                    'threshold': min_feedback_coverage,
                }
            )

        live_f1 = live_metrics.get('f1')
        if live_metrics['sample_size'] < min_live_label_sample:
            alerts.append(
                {
                    'type': 'limited_ground_truth_sample',
                    'severity': 'medium',
                    'message': 'Live performance is based on too few confirmed outcomes to support a strong retraining decision.',
                    'current_value': float(live_metrics['sample_size']),
                    'baseline_value': float(min_live_label_sample),
                    'threshold': float(min_live_label_sample),
                }
            )
        elif live_f1 is not None and not np.isnan(live_f1):
            f1_drop = baseline_metrics['f1'] - live_f1
            if f1_drop > 0.05:
                alerts.append(
                    {
                        'type': 'live_f1_drop',
                        'severity': 'high',
                        'message': f'Live F1 dropped by {f1_drop:.3f} against the production baseline.',
                        'current_value': float(live_f1),
                        'baseline_value': float(baseline_metrics['f1']),
                        'threshold': 0.05,
                    }
                )

    return alerts


def build_retraining_recommendation(alerts: list[dict]) -> dict:
    high_count = sum(alert['severity'] == 'high' for alert in alerts)
    medium_count = sum(alert['severity'] == 'medium' for alert in alerts)
    limited_sample = any(alert.get('type') == 'limited_ground_truth_sample' for alert in alerts)

    if high_count >= 1:
        return {
            'status': 'Retraining Recommended',
            'reason': 'A high-severity monitoring alert was triggered, so the production model should be reviewed and retrained soon.',
        }
    if limited_sample:
        return {
            'status': 'Monitor Closely',
            'reason': 'Live performance signals are still based on a small confirmed-outcome sample, so the system is waiting for stronger evidence before recommending retraining.',
        }
    if medium_count >= 2:
        return {
            'status': 'Monitor Closely',
            'reason': 'Several medium-severity alerts are active. The system should keep collecting evidence and prepare a candidate retraining run.',
        }
    return {
        'status': 'Healthy',
        'reason': 'No major monitoring issue suggests retraining right now.',
    }


def evaluate_governance_candidate(candidate_metrics: dict, production_metrics: dict | None = None) -> dict:
    production_metrics = production_metrics or PRODUCTION_BASELINE_METRICS
    candidate_f1 = candidate_metrics['f1']
    candidate_roc_auc = candidate_metrics['roc_auc']
    production_f1 = production_metrics['f1']
    production_roc_auc = production_metrics['roc_auc']

    if candidate_f1 >= production_f1 + 0.01 and candidate_roc_auc >= production_roc_auc:
        return {
            'decision': 'approve_candidate',
            'rationale': 'Candidate exceeds the production baseline on F1 and matches or exceeds ROC-AUC.',
        }
    if candidate_f1 >= production_f1 and candidate_roc_auc >= production_roc_auc - 0.01:
        return {
            'decision': 'review_candidate',
            'rationale': 'Candidate is competitive with production but should be reviewed before promotion.',
        }
    return {
        'decision': 'reject_candidate',
        'rationale': 'Candidate does not clearly outperform the current production baseline.',
    }


def build_lifecycle_frame(
    alerts: list[dict],
    governance_decision: str | None,
    feedback_coverage: float,
) -> pd.DataFrame:
    alert_count = len(alerts)
    retraining_status = build_retraining_recommendation(alerts)['status']

    lifecycle_rows = [
        {'step': 'Experiment', 'status': 'Completed', 'description': 'Candidate models are tracked in MLflow.'},
        {'step': 'Governance', 'status': governance_decision or 'Pending', 'description': 'Candidate promotion is reviewed against the production baseline.'},
        {'step': 'Production', 'status': 'Active', 'description': 'The approved C=100 logistic regression model serves predictions.'},
        {'step': 'Monitoring', 'status': f'{alert_count} alert(s)', 'description': 'Postgres prediction logs feed drift and risk monitoring.'},
        {'step': 'Ground Truth', 'status': f'{feedback_coverage:.0%} coverage', 'description': 'Delayed outcomes are joined back to predictions for live evaluation.'},
        {'step': 'Retraining', 'status': retraining_status, 'description': 'Monitoring signals drive the retraining recommendation.'},
    ]
    return pd.DataFrame(lifecycle_rows)


def calculate_model_trust_score(alerts: list[dict]) -> int:
    """Calculate a trust score (0-100) based on active alerts."""
    if not alerts:
        return 100
    
    high_count = sum(1 for a in alerts if a.get('severity') == 'high')
    medium_count = sum(1 for a in alerts if a.get('severity') == 'medium')
    
    # Start at 100, deduct for alerts
    score = 100 - (high_count * 25) - (medium_count * 10)
    return max(0, min(100, score))


def run_lab_experiment_cached(params: dict):
    """Run a lab experiment and return results (cached for Streamlit)."""
    from functools import lru_cache
    import json

    cache_key = json.dumps({'params': params}, sort_keys=True, default=str)

    @lru_cache(maxsize=1)
    def _run_experiment(key):
        df = load_data()
        datasets = prepare_modeling_datasets(df)

        result = run_logistic_regression_experiment(datasets=datasets, **params)
        return result

    return _run_experiment(cache_key)


def log_lab_run_to_mlflow(result: dict) -> str:
    """Log a lab experiment result to MLflow."""
    import mlflow
    
    mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    try:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        experiment_name = get_mlflow_experiment_name()

        if mlflow.get_experiment_by_name(experiment_name) is None:
            mlflow.create_experiment(experiment_name)

        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run(run_name=build_lab_run_name(result['params'])) as run:
            mlflow.set_tag('model_type', result['params'].get('model_family', 'logistic_regression'))
            # Log parameters
            for key, value in result['params'].items():
                mlflow.log_param(key, value)
            
            # Log metrics
            for key, value in result['metrics']['test'].items():
                mlflow.log_metric(f'test_{key}', value)
            
            # Log model
            mlflow.sklearn.log_model(result['model'], 'model')
        
        return run.info.run_id
    except Exception as e:
        print(f"MLflow logging failed: {e}")
        raise


def build_lab_run_name(params: dict) -> str:
    """Build a descriptive name for a lab run."""
    c = params.get('C', 'unknown')
    solver = params.get('solver', 'unknown')
    penalty = params.get('penalty', 'unknown')
    return f"lab_C{c}_{solver}_{penalty}"


def get_mlflow_tracking_uri() -> str:
    """Get the MLflow tracking URI."""
    return os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')


def get_mlflow_experiment_name() -> str:
    """Get the MLflow experiment name."""
    return os.getenv('MLFLOW_EXPERIMENT_NAME', 'lr')
