import os
import sys
import logging
from datetime import datetime, timezone
from pathlib import Path

# Suppress MLFlow Git warning before imports
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
logging.getLogger("mlflow.utils.git_utils").setLevel(logging.ERROR)

import pandas as pd
import mlflow
import mlflow.sklearn

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from Scripts.model_utils import (
    CANDIDATE_MODEL_ALIAS,
    MODEL_BUNDLE_PATH,
    MODEL_REGISTRY_NAME,
    PRODUCTION_MODEL_ALIAS,
    apply_feature_engineering,
    calculate_model_selection_score,
    drop_unused_columns,
    encode_target,
    get_latest_model_version_by_alias_or_stage,
    load_data,
    log_bundle_artifact_to_mlflow,
    prepare_features,
    run_logistic_regression_experiment,
    save_bundle,
    scale_features,
    split_data,
)

DATA_PATH = ROOT_DIR / 'telco_churn_cleaned.csv'
BUNDLE_PATH = MODEL_BUNDLE_PATH
DROP_COLUMNS = ['Support_Tickets', 'App_Logins']


def generate_logistic_variants(profile: str = 'quick'):
    variants = []
    profile = profile.lower()

    if profile == 'full':
        variant_specs = [
            {'solver': 'liblinear', 'penalty': 'l1'},
            {'solver': 'liblinear', 'penalty': 'l2'},
            {'solver': 'lbfgs', 'penalty': 'l2'},
            {'solver': 'saga', 'penalty': 'l1'},
            {'solver': 'saga', 'penalty': 'l2'},
            {'solver': 'saga', 'penalty': 'elasticnet'},
        ]
        c_values = [0.003, 0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1, 2, 5, 10, 20, 50, 100]
        class_weight_options = [None, 'balanced', {0: 1, 1: 2}, {0: 1, 1: 3}, {0: 1, 1: 4}]
    elif profile == 'balanced':
        variant_specs = [
            {'solver': 'liblinear', 'penalty': 'l1'},
            {'solver': 'liblinear', 'penalty': 'l2'},
            {'solver': 'lbfgs', 'penalty': 'l2'},
            {'solver': 'saga', 'penalty': 'l2'},
        ]
        c_values = [0.01, 0.1, 1, 10, 100]
        class_weight_options = [None, 'balanced', {0: 1, 1: 2}]
    else:
        variant_specs = [
            {'solver': 'liblinear', 'penalty': 'l1'},
            {'solver': 'liblinear', 'penalty': 'l2'},
            {'solver': 'lbfgs', 'penalty': 'l2'},
        ]
        c_values = [0.1, 1, 10]
        class_weight_options = [None, 'balanced']

    for spec in variant_specs:
        for c_value in c_values:
            for class_weight in class_weight_options:
                l1_ratios = [None]
                if spec['penalty'] == 'elasticnet':
                    l1_ratios = [0.15, 0.5, 0.85]

                for l1_ratio in l1_ratios:
                    class_weight_name = (
                        class_weight
                        if isinstance(class_weight, str) or class_weight is None
                        else f"w{class_weight[1]}"
                    )
                    variant_name = f"lr_{spec['solver']}_{spec['penalty']}_C{c_value}_cw{class_weight_name or 'none'}"
                    if l1_ratio is not None:
                        variant_name += f"_l1r{l1_ratio}"
                    variants.append(
                        {
                            'variant_name': variant_name,
                            'solver': spec['solver'],
                            'penalty': spec['penalty'],
                            'C': c_value,
                            'class_weight': class_weight,
                            'l1_ratio': l1_ratio,
                            'max_iter': 3000,
                        }
                    )
    return variants


def calculate_business_metrics(model, x_test, y_test, df_test):
    """Calculate business-specific KPIs beyond standard ML metrics."""
    y_pred_proba = model.predict_proba(x_test)[:, 1]
    decision_threshold = getattr(model, 'decision_threshold_', 0.5)
    y_pred = (y_pred_proba >= decision_threshold).astype(int)

    high_risk_predictions = (y_pred_proba >= 0.7).sum()
    medium_risk_predictions = ((y_pred_proba >= 0.5) & (y_pred_proba < 0.7)).sum()
    low_risk_predictions = (y_pred_proba < 0.5).sum()

    avg_customer_value = 1000
    potential_loss_prevented = high_risk_predictions * avg_customer_value * 0.3

    tenure_groups = pd.cut(
        df_test['tenure'],
        bins=[0, 12, 24, 48, 72],
        labels=['New', 'Growing', 'Established', 'Loyal'],
    )
    churn_by_segment = {}
    for segment in tenure_groups.unique():
        segment_mask = tenure_groups == segment
        if segment_mask.sum() > 0:
            churn_rate = y_test[segment_mask].mean()
            churn_by_segment[f'{segment}_churn_rate'] = churn_rate

    return {
        'total_predictions': len(y_pred),
        'high_risk_predictions': high_risk_predictions,
        'high_risk_rate': high_risk_predictions / len(y_pred) if len(y_pred) else 0.0,
        'medium_risk_predictions': medium_risk_predictions,
        'low_risk_predictions': low_risk_predictions,
        'potential_loss_prevented': potential_loss_prevented,
        'prediction_coverage': len(y_pred) / len(df_test) * 100,
        **churn_by_segment,
    }


def assert_run_artifacts_present(client, run_id: str, required_paths: tuple[str, ...] = ('production_bundle',)):
    """Fail fast when required non-model run artifacts were not persisted."""
    artifacts = client.list_artifacts(run_id)
    available_paths = {getattr(artifact, 'path', '') for artifact in artifacts}
    missing_paths = [path for path in required_paths if path not in available_paths]
    if missing_paths:
        raise RuntimeError(
            f'MLflow run {run_id} is missing required artifacts: {", ".join(missing_paths)}. '
            f'Available top-level artifacts: {sorted(path for path in available_paths if path)}'
        )


def log_model_with_compatible_uri(model):
    """Log a model in a way that works with modern MLflow model URIs."""
    model_info = mlflow.sklearn.log_model(
        model,
        name='model',
    )
    model_uri = getattr(model_info, 'model_uri', None)
    if not model_uri:
        raise RuntimeError('MLflow did not return a model URI after logging the production model.')
    return model_info, model_uri


def register_best_model(best_model_record, dv, scaler):
    """Register the best model in MLflow and log a Streamlit-ready production bundle."""
    try:
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'))
        client = mlflow.MlflowClient()

        versions = client.search_model_versions(f"name = '{MODEL_REGISTRY_NAME}'")
        if versions:
            latest_version = max(versions, key=lambda version: int(version.version))
            current_version = int(latest_version.version)
        else:
            current_version = 0

        new_version = current_version + 1

        with mlflow.start_run(run_name=f'production_v{new_version}') as run:
            mlflow.set_tag('model_type', best_model_record['params'].get('model_family', 'logistic_regression'))
            mlflow.set_tag('stage', 'production')
            mlflow.set_tag('best_model', 'true')
            mlflow.log_param('production_version', new_version)
            mlflow.log_param('selected_by', 'validation_f1')
            mlflow.log_param('validation_f1', best_model_record['val_metrics']['f1'])
            mlflow.log_metrics(best_model_record['business_metrics'])
            mlflow.log_metrics({f'val_{k}': v for k, v in best_model_record['val_metrics'].items()})
            mlflow.log_metrics({f'test_{k}': v for k, v in best_model_record['test_metrics'].items()})

            production_bundle_metadata = {
                'model_family': best_model_record['params'].get('model_family', 'logistic_regression'),
                'variant_name': best_model_record['name'],
                'decision_threshold': best_model_record['params']['decision_threshold'],
                'source_candidate_run_id': best_model_record['run_id'],
                'deployed_at': datetime.now(timezone.utc).isoformat(),
                'selection_score': calculate_model_selection_score(best_model_record['val_metrics']),
                'val_metrics': best_model_record['val_metrics'],
                'test_metrics': best_model_record['test_metrics'],
            }
            _, model_uri = log_model_with_compatible_uri(best_model_record['model'])
            log_bundle_artifact_to_mlflow(
                best_model_record['model'],
                dv,
                scaler,
                metadata=production_bundle_metadata,
            )
            assert_run_artifacts_present(client, run.info.run_id)

            model_version = mlflow.register_model(model_uri, MODEL_REGISTRY_NAME)
            client.set_model_version_tag(MODEL_REGISTRY_NAME, model_version.version, 'deployment_status', 'candidate')
            client.set_registered_model_alias(MODEL_REGISTRY_NAME, CANDIDATE_MODEL_ALIAS, model_version.version)

            if new_version == 1:
                client.set_registered_model_alias(MODEL_REGISTRY_NAME, PRODUCTION_MODEL_ALIAS, model_version.version)
                client.set_model_version_tag(MODEL_REGISTRY_NAME, model_version.version, 'deployment_status', 'production')
            else:
                try:
                    prod_version = get_latest_model_version_by_alias_or_stage(
                        client,
                        MODEL_REGISTRY_NAME,
                        alias=PRODUCTION_MODEL_ALIAS,
                        stage='Production',
                    )
                    if prod_version:
                        prod_run = mlflow.get_run(prod_version.run_id)
                        prod_f1 = prod_run.data.metrics.get('val_f1', 0)

                        if best_model_record['val_metrics']['f1'] > prod_f1:
                            client.set_registered_model_alias(MODEL_REGISTRY_NAME, PRODUCTION_MODEL_ALIAS, model_version.version)
                            client.set_model_version_tag(MODEL_REGISTRY_NAME, model_version.version, 'deployment_status', 'production')
                            print(
                                f'New model version {model_version.version} promoted to production '
                                f'(F1: {best_model_record["val_metrics"]["f1"]:.4f} > {prod_f1:.4f})'
                            )
                        else:
                            client.set_model_version_tag(MODEL_REGISTRY_NAME, model_version.version, 'deployment_status', 'candidate')
                            print(
                                f'New model version {model_version.version} kept as candidate '
                                f'(F1: {best_model_record["val_metrics"]["f1"]:.4f} <= {prod_f1:.4f})'
                            )
                except Exception as error:
                    print(f'Could not compare with production model: {error}')
                    client.set_model_version_tag(MODEL_REGISTRY_NAME, model_version.version, 'deployment_status', 'candidate')

            print(f'Model registered as version {model_version.version} of "{MODEL_REGISTRY_NAME}"')

    except Exception as error:
        raise RuntimeError(f'Could not register model: {error}') from error


def main(profile: str = 'quick'):
    df = load_data(DATA_PATH)
    df_train, df_val, df_test = split_data(df)

    y_train = encode_target(df_train['Churn'])
    y_val = encode_target(df_val['Churn'])
    y_test = encode_target(df_test['Churn'])

    df_train = df_train.drop(columns=['Churn'])
    df_val = df_val.drop(columns=['Churn'])
    df_test = df_test.drop(columns=['Churn'])

    df_train_fe = apply_feature_engineering(df_train)
    df_val_fe = apply_feature_engineering(df_val)
    df_test_fe = apply_feature_engineering(df_test)

    df_train_final = drop_unused_columns(df_train_fe, DROP_COLUMNS)
    df_val_final = drop_unused_columns(df_val_fe, DROP_COLUMNS)
    df_test_final = drop_unused_columns(df_test_fe, DROP_COLUMNS)

    x_train, dv = prepare_features(df_train_final, fit_dv=True)
    x_val, _ = prepare_features(df_val_final, dv=dv)
    x_test, _ = prepare_features(df_test_final, dv=dv)

    x_train_scaled, scaler = scale_features(x_train)
    x_val_scaled, _ = scale_features(x_val, scaler=scaler)
    x_test_scaled, _ = scale_features(x_test, scaler=scaler)

    variants = generate_logistic_variants(profile=profile)
    print(f'Running training profile "{profile}" with {len(variants)} variants')

    mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    mlflow_experiment = os.getenv('MLFLOW_EXPERIMENT_NAME', 'churn_prediction')
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment)

    all_results = []
    for variant in variants:
        params = {k: v for k, v in variant.items() if k != 'variant_name'}
        result = run_logistic_regression_experiment(
            datasets={
                'dv': dv,
                'scaler': scaler,
                'train': {'X': x_train_scaled, 'y': y_train, 'features': df_train_final},
                'val': {'X': x_val_scaled, 'y': y_val, 'features': df_val_final},
                'test': {'X': x_test_scaled, 'y': y_test, 'features': df_test_final},
            },
            **params,
        )

        model = result['model']
        params = result['params']
        train_metrics = result['metrics']['train']
        val_metrics = result['metrics']['val']
        test_metrics = result['metrics']['test']

        with mlflow.start_run(run_name=variant['variant_name']) as run:
            mlflow.set_tag('model_type', 'logistic_regression')
            mlflow.log_param('model_name', 'logistic_regression')
            mlflow.log_param('variant_name', variant['variant_name'])
            mlflow.log_params(params)
            mlflow.log_metric('val_selection_score', calculate_model_selection_score(val_metrics))
            mlflow.log_metrics({f'train_{k}': v for k, v in train_metrics.items()})
            mlflow.log_metrics({f'val_{k}': v for k, v in val_metrics.items()})
            mlflow.log_metrics({f'test_{k}': v for k, v in test_metrics.items()})
            mlflow.log_param('experiment_name', mlflow_experiment)
            mlflow.sklearn.log_model(model, name='model', registered_model_name=MODEL_REGISTRY_NAME)

            business_metrics = calculate_business_metrics(model, x_test_scaled, y_test, df_test_final)
            mlflow.log_metrics(business_metrics)

            all_results.append(
                {
                    'name': variant['variant_name'],
                    'params': params,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'test_metrics': test_metrics,
                    'business_metrics': business_metrics,
                    'model': model,
                    'run_id': run.info.run_id,
                }
            )
            print(f'Logged MLflow run for {variant["variant_name"]}: {run.info.run_id}')

    best_model_record = max(
        all_results,
        key=lambda record: (
            calculate_model_selection_score(record['val_metrics']),
            record['val_metrics']['f1'],
            record['val_metrics']['roc_auc'],
            -record['val_metrics']['log_loss'],
        ),
    )
    bundle_metadata = {
        'model_family': best_model_record['params'].get('model_family', 'logistic_regression'),
        'variant_name': best_model_record['name'],
        'decision_threshold': best_model_record['params']['decision_threshold'],
        'run_id': best_model_record['run_id'],
        'deployed_at': datetime.now(timezone.utc).isoformat(),
        'selection_score': calculate_model_selection_score(best_model_record['val_metrics']),
        'val_metrics': best_model_record['val_metrics'],
        'test_metrics': best_model_record['test_metrics'],
    }
    save_bundle(BUNDLE_PATH, best_model_record['model'], dv, scaler, metadata=bundle_metadata)

    register_best_model(best_model_record, dv, scaler)

    print(f'Best model saved to: {BUNDLE_PATH} ({best_model_record["name"]})')


if __name__ == '__main__':
    main()
