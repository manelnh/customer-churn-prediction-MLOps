from datetime import datetime, timezone
import unittest
from unittest.mock import patch

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler

from Scripts import model_utils


class ModelUtilsTests(unittest.TestCase):
    def test_apply_feature_engineering_adds_expected_columns(self):
        df = pd.DataFrame(
            [
                {
                    'tenure': 10,
                    'MonthlyCharges': 50.0,
                    'Data_Usage_GB': 20.0,
                    'OnlineSecurity': 'Yes',
                    'OnlineBackup': 'No',
                    'DeviceProtection': 'Yes',
                    'StreamingTV': 'No',
                    'StreamingMovies': 'Yes',
                    'SeniorCitizen': 'No',
                }
            ]
        )

        engineered = model_utils.apply_feature_engineering(df)

        self.assertIn('TotalServices', engineered.columns)
        self.assertIn('Total_Revenue', engineered.columns)
        self.assertIn('Monthly_per_Tenure', engineered.columns)
        self.assertIn('Is_First_Year', engineered.columns)
        self.assertIn('Usage_per_Month', engineered.columns)
        self.assertEqual(engineered.loc[0, 'TotalServices'], 3)
        self.assertEqual(engineered.loc[0, 'Total_Revenue'], 500.0)

    def test_prepare_customer_features_returns_scaled_row(self):
        training_rows = [
            {
                'tenure': 6,
                'MonthlyCharges': 65.0,
                'Data_Usage_GB': 30.0,
                'OnlineSecurity': 'Yes',
                'OnlineBackup': 'No',
                'DeviceProtection': 'Yes',
                'StreamingTV': 'No',
                'StreamingMovies': 'No',
                'SeniorCitizen': 'No',
                'Support_Tickets': 2,
                'App_Logins': 12,
            },
            {
                'tenure': 24,
                'MonthlyCharges': 90.0,
                'Data_Usage_GB': 80.0,
                'OnlineSecurity': 'No',
                'OnlineBackup': 'Yes',
                'DeviceProtection': 'Yes',
                'StreamingTV': 'Yes',
                'StreamingMovies': 'Yes',
                'SeniorCitizen': 'Yes',
                'Support_Tickets': 5,
                'App_Logins': 8,
            },
        ]
        train_df = model_utils.drop_unused_columns(
            model_utils.apply_feature_engineering(pd.DataFrame(training_rows))
        )
        dv = DictVectorizer(sparse=False)
        X = dv.fit_transform(train_df.to_dict(orient='records'))
        scaler = StandardScaler().fit(X)

        customer = training_rows[0]
        X_scaled, final_df = model_utils.prepare_customer_features(customer, dv, scaler)

        self.assertEqual(X_scaled.shape, (1, len(dv.get_feature_names_out())))
        self.assertNotIn('Support_Tickets', final_df.columns)
        self.assertNotIn('App_Logins', final_df.columns)
        self.assertIn('Total_Revenue', final_df.columns)

    def test_calculate_live_metrics_handles_empty_ground_truth(self):
        logs = pd.DataFrame(
            [
                {
                    'predicted_label': 'Churn',
                    'predicted_probability': 0.8,
                    'actual_label': None,
                }
            ]
        )

        metrics = model_utils.calculate_live_metrics(logs)

        self.assertEqual(metrics['coverage'], 0.0)
        self.assertEqual(metrics['sample_size'], 0)

    def test_determine_best_threshold_returns_valid_threshold(self):
        y_true = [0, 1, 1, 0]
        probabilities = [0.10, 0.62, 0.87, 0.45]

        threshold, best_f1 = model_utils.determine_best_threshold(y_true, probabilities)

        self.assertGreaterEqual(threshold, 0.20)
        self.assertLessEqual(threshold, 0.80)
        self.assertGreater(best_f1, 0.0)

    def test_calculate_model_selection_score_rewards_stronger_candidate(self):
        strong_metrics = {'accuracy': 0.88, 'f1': 0.81, 'roc_auc': 0.94, 'log_loss': 0.31}
        weak_metrics = {'accuracy': 0.86, 'f1': 0.76, 'roc_auc': 0.91, 'log_loss': 0.39}

        strong_score = model_utils.calculate_model_selection_score(strong_metrics)
        weak_score = model_utils.calculate_model_selection_score(weak_metrics)

        self.assertGreater(strong_score, weak_score)


    def test_detect_monitoring_alerts_flags_low_volume_and_limited_sample(self):
        base_time = datetime(2026, 4, 30, tzinfo=timezone.utc)
        logs = pd.DataFrame(
            [
                {
                    'created_at': base_time,
                    'predicted_probability': 0.81,
                    'predicted_risk': 'HIGH RISK',
                    'predicted_label': 'Churn',
                    'actual_label': 'Churn',
                },
                {
                    'created_at': base_time,
                    'predicted_probability': 0.77,
                    'predicted_risk': 'HIGH RISK',
                    'predicted_label': 'Churn',
                    'actual_label': None,
                },
                {
                    'created_at': base_time,
                    'predicted_probability': 0.32,
                    'predicted_risk': 'LOW RISK',
                    'predicted_label': 'No churn',
                    'actual_label': None,
                },
                {
                    'created_at': base_time,
                    'predicted_probability': 0.68,
                    'predicted_risk': 'MEDIUM RISK',
                    'predicted_label': 'Churn',
                    'actual_label': None,
                },
                {
                    'created_at': base_time,
                    'predicted_probability': 0.41,
                    'predicted_risk': 'LOW RISK',
                    'predicted_label': 'No churn',
                    'actual_label': None,
                },
            ]
        )

        alerts = model_utils.detect_monitoring_alerts(logs)
        alert_types = {alert['type'] for alert in alerts}

        self.assertIn('low_prediction_volume', alert_types)
        self.assertIn('ground_truth_gap', alert_types)
        self.assertIn('limited_ground_truth_sample', alert_types)

    def test_load_model_bundle_prefers_mlflow_production_bundle(self):
        dv = DictVectorizer(sparse=False)
        x = dv.fit_transform([{'tenure': 1, 'MonthlyCharges': 10.0}, {'tenure': 2, 'MonthlyCharges': 20.0}])
        scaler = StandardScaler().fit(x)

        class DummyModel:
            n_features_in_ = x.shape[1]

        bundle = {'model': DummyModel(), 'dv': dv, 'scaler': scaler, 'metadata': {}}

        with patch.object(model_utils, 'mlflow', object()), \
                patch.object(model_utils, 'load_latest_production_bundle_from_mlflow', return_value=bundle), \
                patch.object(model_utils, 'load_bundle') as load_bundle_mock:
            model, loaded_dv, loaded_scaler = model_utils.load_model_bundle()

        self.assertIs(model, bundle['model'])
        self.assertIs(loaded_dv, dv)
        self.assertIs(loaded_scaler, scaler)
        load_bundle_mock.assert_not_called()

    def test_get_latest_model_version_by_alias_or_stage_prefers_alias(self):
        client = unittest.mock.Mock()
        alias_version = unittest.mock.Mock(version='7', run_id='run-7')
        client.get_model_version_by_alias.return_value = alias_version

        result = model_utils.get_latest_model_version_by_alias_or_stage(client, 'lr')

        self.assertIs(result, alias_version)
        client.search_model_versions.assert_not_called()

    def test_get_latest_model_version_by_alias_or_stage_falls_back_to_stage(self):
        client = unittest.mock.Mock()
        client.get_model_version_by_alias.side_effect = RuntimeError('alias missing')
        client.search_model_versions.return_value = [
            unittest.mock.Mock(version='2', current_stage='Staging'),
            unittest.mock.Mock(version='5', current_stage='Production'),
            unittest.mock.Mock(version='4', current_stage='Production'),
        ]

        result = model_utils.get_latest_model_version_by_alias_or_stage(client, 'lr')

        self.assertEqual(result.version, '5')

    def test_load_model_bundle_falls_back_to_local_bundle_when_mlflow_fails(self):
        dv = DictVectorizer(sparse=False)
        x = dv.fit_transform([{'tenure': 1, 'MonthlyCharges': 10.0}, {'tenure': 2, 'MonthlyCharges': 20.0}])
        scaler = StandardScaler().fit(x)

        class DummyModel:
            n_features_in_ = x.shape[1]

        local_bundle = {'model': DummyModel(), 'dv': dv, 'scaler': scaler, 'metadata': {}}

        with patch.object(model_utils, 'mlflow', object()), \
                patch.object(
                    model_utils,
                    'load_latest_production_bundle_from_mlflow',
                    side_effect=RuntimeError('registry unavailable'),
                ), \
                patch.object(model_utils, 'load_bundle', return_value=local_bundle) as load_bundle_mock:
            model, loaded_dv, loaded_scaler = model_utils.load_model_bundle()

        self.assertIs(model, local_bundle['model'])
        self.assertIs(loaded_dv, dv)
        self.assertIs(loaded_scaler, scaler)
        load_bundle_mock.assert_called()

    def test_get_baseline_metrics_from_metadata_prefers_bundle_metrics(self):
        metadata = {
            'test_metrics': {
                'accuracy': 0.91,
                'f1': 0.83,
                'roc_auc': 0.95,
            }
        }

        baseline = model_utils.get_baseline_metrics_from_metadata(metadata)

        self.assertEqual(baseline['accuracy'], 0.91)
        self.assertEqual(baseline['f1'], 0.83)
        self.assertEqual(baseline['roc_auc'], 0.95)

    def test_filter_predictions_for_active_model_prefers_mlflow_run_id(self):
        logs = pd.DataFrame(
            [
                {'id': 1, 'mlflow_run_id': 'run_new', 'model_version': 'new_variant'},
                {'id': 2, 'mlflow_run_id': 'run_old', 'model_version': 'old_variant'},
            ]
        )

        filtered = model_utils.filter_predictions_for_active_model(
            logs,
            {'mlflow_run_id': 'run_new', 'variant_name': 'new_variant'},
        )

        self.assertEqual(filtered['id'].tolist(), [1])

    def test_filter_predictions_for_active_model_falls_back_to_variant_name(self):
        logs = pd.DataFrame(
            [
                {'id': 1, 'mlflow_run_id': None, 'model_version': 'new_variant'},
                {'id': 2, 'mlflow_run_id': None, 'model_version': 'old_variant'},
            ]
        )

        filtered = model_utils.filter_predictions_for_active_model(
            logs,
            {'variant_name': 'new_variant'},
        )

        self.assertEqual(filtered['id'].tolist(), [1])

    def test_filter_predictions_for_active_model_prefers_post_deployment_window(self):
        logs = pd.DataFrame(
            [
                {'id': 1, 'created_at': '2026-05-01T10:00:00Z', 'mlflow_run_id': 'run_new', 'model_version': 'new_variant'},
                {'id': 2, 'created_at': '2026-05-03T10:00:00Z', 'mlflow_run_id': 'run_new', 'model_version': 'new_variant'},
                {'id': 3, 'created_at': '2026-05-04T10:00:00Z', 'mlflow_run_id': 'run_new', 'model_version': 'new_variant'},
            ]
        )

        filtered = model_utils.filter_predictions_for_active_model(
            logs,
            {'mlflow_run_id': 'run_new', 'variant_name': 'new_variant', 'deployed_at': '2026-05-03T00:00:00Z'},
        )

        self.assertEqual(filtered['id'].tolist(), [2, 3])


if __name__ == '__main__':
    unittest.main()
