import os
from typing import Any

import pandas as pd
import psycopg2
from psycopg2.extras import Json


def get_database_url() -> str:
    return os.getenv(
        'DATABASE_URL',
        (
            f"postgresql+psycopg2://"
            f"{os.getenv('POSTGRES_USER', 'postgres')}:"
            f"{os.getenv('POSTGRES_PASSWORD', 'postgres')}@"
            f"{os.getenv('POSTGRES_HOST', 'db')}:"
            f"{os.getenv('POSTGRES_PORT', '5432')}/"
            f"{os.getenv('POSTGRES_DB', 'churn_db')}"
        ),
    )


def get_postgres_connection():
    database_url = os.getenv('DATABASE_URL')
    if database_url:
        sanitized_url = database_url.replace('postgresql+psycopg2://', 'postgresql://', 1)
        return psycopg2.connect(sanitized_url)

    return psycopg2.connect(
        host=os.getenv('POSTGRES_HOST', 'db'),
        port=int(os.getenv('POSTGRES_PORT', 5432)),
        dbname=os.getenv('POSTGRES_DB', 'churn_db'),
        user=os.getenv('POSTGRES_USER', 'postgres'),
        password=os.getenv('POSTGRES_PASSWORD', 'postgres'),
    )


def ensure_prediction_table(connection):
    with connection.cursor() as cursor:
        cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS prediction_logs (
                id SERIAL PRIMARY KEY,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                model_name TEXT DEFAULT 'logistic_regression',
                model_version TEXT DEFAULT 'C100_lbfgs_l2',
                model_stage TEXT DEFAULT 'Production',
                source TEXT,
                mlflow_run_id TEXT,
                input_features JSONB,
                top_drivers JSONB,
                predicted_probability DOUBLE PRECISION,
                predicted_label TEXT,
                predicted_risk TEXT
            );
            '''
        )
        cursor.execute(
            '''
            ALTER TABLE prediction_logs
            ADD COLUMN IF NOT EXISTS model_name TEXT DEFAULT 'logistic_regression',
            ADD COLUMN IF NOT EXISTS model_version TEXT DEFAULT 'C100_lbfgs_l2',
            ADD COLUMN IF NOT EXISTS model_stage TEXT DEFAULT 'Production',
            ADD COLUMN IF NOT EXISTS source TEXT,
            ADD COLUMN IF NOT EXISTS mlflow_run_id TEXT,
            ADD COLUMN IF NOT EXISTS input_features JSONB,
            ADD COLUMN IF NOT EXISTS top_drivers JSONB,
            ADD COLUMN IF NOT EXISTS predicted_probability DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS predicted_label TEXT,
            ADD COLUMN IF NOT EXISTS predicted_risk TEXT,
            ADD COLUMN IF NOT EXISTS actual_label TEXT,
            ADD COLUMN IF NOT EXISTS actual_risk TEXT,
            ADD COLUMN IF NOT EXISTS actual_label_at TIMESTAMPTZ,
            ADD COLUMN IF NOT EXISTS ground_truth_source TEXT,
            ADD COLUMN IF NOT EXISTS feedback_notes TEXT,
            ADD COLUMN IF NOT EXISTS manager_action TEXT,
            ADD COLUMN IF NOT EXISTS manager_action_at TIMESTAMPTZ;
            '''
        )
        connection.commit()


def ensure_monitoring_alerts_table(connection):
    with connection.cursor() as cursor:
        cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS monitoring_alerts (
                id SERIAL PRIMARY KEY,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                message TEXT NOT NULL,
                current_value DOUBLE PRECISION,
                baseline_value DOUBLE PRECISION,
                threshold_value DOUBLE PRECISION,
                status TEXT NOT NULL DEFAULT 'open',
                resolved_at TIMESTAMPTZ,
                details JSONB
            );
            '''
        )
        connection.commit()


def ensure_governance_table(connection):
    with connection.cursor() as cursor:
        cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS governance_decisions (
                id SERIAL PRIMARY KEY,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                candidate_name TEXT NOT NULL,
                mlflow_run_id TEXT,
                candidate_source TEXT NOT NULL,
                decision TEXT NOT NULL,
                rationale TEXT NOT NULL,
                approver TEXT,
                candidate_params JSONB,
                candidate_metrics JSONB,
                production_metrics JSONB
            );
            '''
        )
        connection.commit()

def ensure_platform_tables(connection):
    ensure_prediction_table(connection)
    ensure_monitoring_alerts_table(connection)
    ensure_governance_table(connection)

def bootstrap_platform_tables_if_enabled(connection) -> bool:
    """
    Create/ensure all platform tables (predictions, monitoring alerts,
    governance decisions), but only if bootstrapping is enabled via env var.

    This lets you disable ad-hoc table creation in environments where
    Alembic migrations are the source of truth for schema changes
    (e.g. production), while still allowing it for local/dev convenience.
    """
    if os.getenv('BOOTSTRAP_TABLES_ON_START', 'true').lower() not in ('1', 'true', 'yes'):
        return False

    ensure_platform_tables(connection)
    return True


def bootstrap_platform_tables_if_enabled(connection) -> bool:
    """Optionally create platform tables for local bootstrap scenarios.

    Production-style environments should apply Alembic migrations explicitly.
    This fallback is only enabled when ENABLE_RUNTIME_DB_BOOTSTRAP is set.
    """
    enabled = os.getenv('ENABLE_RUNTIME_DB_BOOTSTRAP', '').strip().lower()
    if enabled in {'1', 'true', 'yes', 'on'}:
        ensure_platform_tables(connection)
        return True
    return False


def insert_prediction_log(
    connection,
    input_features: dict,
    predicted_probability: float,
    predicted_label: str,
    predicted_risk: str,
    top_drivers: list[dict[str, Any]] | None = None,
    model_name: str = 'logistic_regression',
    model_version: str = 'C100_lbfgs_l2',
    model_stage: str = 'Production',
    source: str = 'streamlit_app',
    mlflow_run_id: str | None = None,
) -> int:
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                '''
                INSERT INTO prediction_logs (
                    model_name,
                    model_version,
                    model_stage,
                    source,
                    mlflow_run_id,
                    input_features,
                    top_drivers,
                    predicted_probability,
                    predicted_label,
                    predicted_risk
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id;
                ''',
                (
                    model_name,
                    model_version,
                    model_stage,
                    source,
                    mlflow_run_id,
                    Json(input_features),
                    Json(top_drivers or []),
                    float(predicted_probability),
                    predicted_label,
                    predicted_risk,
                ),
            )
            prediction_id = cursor.fetchone()[0]
        connection.commit()
        return prediction_id
    except Exception as error:
        connection.rollback()
        raise error


def update_prediction_ground_truth(
    connection,
    prediction_id: int,
    actual_label: str,
    ground_truth_source: str = 'manual_review',
    feedback_notes: str | None = None,
):
    actual_risk = 'HIGH RISK' if actual_label == 'Churn' else 'LOW RISK'
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                '''
                UPDATE prediction_logs
                SET actual_label = %s,
                    actual_risk = %s,
                    actual_label_at = NOW(),
                    ground_truth_source = %s,
                    feedback_notes = %s
                WHERE id = %s;
                ''',
                (
                    actual_label,
                    actual_risk,
                    ground_truth_source,
                    feedback_notes,
                    prediction_id,
                ),
            )
        connection.commit()
    except Exception as error:
        connection.rollback()
        raise error


def update_prediction_manager_action(
    connection,
    prediction_id: int,
    manager_action: str,
):
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                '''
                UPDATE prediction_logs
                SET manager_action = %s,
                    manager_action_at = NOW()
                WHERE id = %s;
                ''',
                (
                    manager_action,
                    prediction_id,
                ),
            )
        connection.commit()
    except Exception as error:
        connection.rollback()
        raise error


def fetch_recent_predictions(connection, limit: int = 50) -> pd.DataFrame:
    with connection.cursor() as cursor:
        cursor.execute(
            '''
            SELECT
                id,
                created_at,
                model_name,
                model_version,
                model_stage,
                source,
                mlflow_run_id,
                input_features,
                top_drivers,
                predicted_probability,
                predicted_label,
                predicted_risk,
                actual_label,
                actual_risk,
                actual_label_at,
                ground_truth_source,
                feedback_notes,
                manager_action,
                manager_action_at
            FROM prediction_logs
            ORDER BY created_at DESC
            LIMIT %s;
            ''',
            (limit,),
        )
        rows = cursor.fetchall()

    records = []
    for row in rows:
        input_features = row[7] or {}
        record = {
            'id': row[0],
            'created_at': row[1],
            'model_name': row[2],
            'model_version': row[3],
            'model_stage': row[4],
            'source': row[5],
            'mlflow_run_id': row[6],
            'top_drivers': row[8] or [],
            'predicted_probability': row[9],
            'predicted_label': row[10],
            'predicted_risk': row[11],
            'actual_label': row[12],
            'actual_risk': row[13],
            'actual_label_at': row[14],
            'ground_truth_source': row[15],
            'feedback_notes': row[16],
            'manager_action': row[17],
            'manager_action_at': row[18],
        }
        if isinstance(input_features, dict):
            record.update(input_features)
        records.append(record)

    return pd.DataFrame(records)


def fetch_all_predictions(connection) -> pd.DataFrame:
    with connection.cursor() as cursor:
        cursor.execute(
            '''
            SELECT
                id,
                created_at,
                model_name,
                model_version,
                model_stage,
                source,
                mlflow_run_id,
                input_features,
                top_drivers,
                predicted_probability,
                predicted_label,
                predicted_risk,
                actual_label,
                actual_risk,
                actual_label_at,
                ground_truth_source,
                feedback_notes,
                manager_action,
                manager_action_at
            FROM prediction_logs
            ORDER BY created_at DESC;
            '''
        )
        rows = cursor.fetchall()

    records = []
    for row in rows:
        input_features = row[7] or {}
        record = {
            'id': row[0],
            'created_at': row[1],
            'model_name': row[2],
            'model_version': row[3],
            'model_stage': row[4],
            'source': row[5],
            'mlflow_run_id': row[6],
            'top_drivers': row[8] or [],
            'predicted_probability': row[9],
            'predicted_label': row[10],
            'predicted_risk': row[11],
            'actual_label': row[12],
            'actual_risk': row[13],
            'actual_label_at': row[14],
            'ground_truth_source': row[15],
            'feedback_notes': row[16],
            'manager_action': row[17],
            'manager_action_at': row[18],
        }
        if isinstance(input_features, dict):
            record.update(input_features)
        records.append(record)

    return pd.DataFrame(records)


def fetch_prediction_totals(connection) -> dict:
    with connection.cursor() as cursor:
        cursor.execute(
            '''
            SELECT predicted_label, COUNT(*)
            FROM prediction_logs
            GROUP BY predicted_label;
            '''
        )
        rows = cursor.fetchall()
    return {label: count for label, count in rows}


def replace_monitoring_alerts(connection, alerts: list[dict[str, Any]]):
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                '''
                UPDATE monitoring_alerts
                SET status = 'resolved',
                    resolved_at = NOW()
                WHERE status = 'open';
                '''
            )

            for alert in alerts:
                cursor.execute(
                    '''
                    INSERT INTO monitoring_alerts (
                        alert_type,
                        severity,
                        message,
                        current_value,
                        baseline_value,
                        threshold_value,
                        details
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s);
                    ''',
                    (
                        alert.get('type'),
                        alert.get('severity'),
                        alert.get('message'),
                        alert.get('current_value'),
                        alert.get('baseline_value'),
                        alert.get('threshold'),
                        Json(alert),
                    ),
                )
        connection.commit()
    except Exception as error:
        connection.rollback()
        raise error


def fetch_recent_alerts(connection, limit: int = 25) -> pd.DataFrame:
    with connection.cursor() as cursor:
        cursor.execute(
            '''
            SELECT
                id,
                created_at,
                alert_type,
                severity,
                message,
                current_value,
                baseline_value,
                threshold_value,
                status,
                resolved_at,
                details
            FROM monitoring_alerts
            ORDER BY created_at DESC
            LIMIT %s;
            ''',
            (limit,),
        )
        rows = cursor.fetchall()

    columns = [
        'id',
        'created_at',
        'alert_type',
        'severity',
        'message',
        'current_value',
        'baseline_value',
        'threshold_value',
        'status',
        'resolved_at',
        'details',
    ]
    return pd.DataFrame(rows, columns=columns)


def insert_governance_decision(
    connection,
    candidate_name: str,
    candidate_source: str,
    decision: str,
    rationale: str,
    candidate_params: dict[str, Any],
    candidate_metrics: dict[str, Any],
    production_metrics: dict[str, Any],
    mlflow_run_id: str | None = None,
    approver: str = 'system_review',
):
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                '''
                INSERT INTO governance_decisions (
                    candidate_name,
                    mlflow_run_id,
                    candidate_source,
                    decision,
                    rationale,
                    approver,
                    candidate_params,
                    candidate_metrics,
                    production_metrics
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
                ''',
                (
                    candidate_name,
                    mlflow_run_id,
                    candidate_source,
                    decision,
                    rationale,
                    approver,
                    Json(candidate_params),
                    Json(candidate_metrics),
                    Json(production_metrics),
                ),
            )
        connection.commit()
    except Exception as error:
        connection.rollback()
        raise error


def fetch_recent_governance_decisions(connection, limit: int = 10) -> pd.DataFrame:
    with connection.cursor() as cursor:
        cursor.execute(
            '''
            SELECT
                id,
                created_at,
                candidate_name,
                mlflow_run_id,
                candidate_source,
                decision,
                rationale,
                approver,
                candidate_params,
                candidate_metrics,
                production_metrics
            FROM governance_decisions
            ORDER BY created_at DESC
            LIMIT %s;
            ''',
            (limit,),
        )
        rows = cursor.fetchall()

    columns = [
        'id',
        'created_at',
        'candidate_name',
        'mlflow_run_id',
        'candidate_source',
        'decision',
        'rationale',
        'approver',
        'candidate_params',
        'candidate_metrics',
        'production_metrics',
    ]
    return pd.DataFrame(rows, columns=columns)
