import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

mlflow_module = types.ModuleType('mlflow')
mlflow_sklearn_module = types.ModuleType('mlflow.sklearn')
mlflow_module.sklearn = mlflow_sklearn_module
mlflow_module.set_tracking_uri = lambda *args, **kwargs: None
mlflow_module.MlflowClient = object
mlflow_module.start_run = lambda *args, **kwargs: None
mlflow_module.set_tag = lambda *args, **kwargs: None
mlflow_module.log_param = lambda *args, **kwargs: None
mlflow_module.log_metrics = lambda *args, **kwargs: None
mlflow_module.get_run = lambda *args, **kwargs: None
mlflow_module.register_model = lambda *args, **kwargs: None
mlflow_sklearn_module.log_model = lambda *args, **kwargs: None
sys.modules['mlflow'] = mlflow_module
sys.modules['mlflow.sklearn'] = mlflow_sklearn_module

from Scripts import train


class TrainTests(unittest.TestCase):
    def test_calculate_business_metrics_includes_high_risk_rate(self):
        class DummyModel:
            decision_threshold_ = 0.5

            @staticmethod
            def predict_proba(x_test):
                return np.array(
                    [
                        [0.10, 0.90],
                        [0.45, 0.55],
                        [0.75, 0.25],
                    ]
                )

        df_test = pd.DataFrame({'tenure': [6, 18, 36]})
        metrics = train.calculate_business_metrics(
            DummyModel(),
            np.array([[1.0], [2.0], [3.0]]),
            np.array([1, 1, 0]),
            df_test,
        )

        self.assertEqual(metrics['total_predictions'], 3)
        self.assertAlmostEqual(metrics['high_risk_rate'], 1 / 3)

    def test_assert_run_artifacts_present_accepts_expected_artifacts(self):
        client = MagicMock()
        client.list_artifacts.return_value = [types.SimpleNamespace(path='production_bundle')]

        train.assert_run_artifacts_present(client, 'production-run')

    def test_assert_run_artifacts_present_raises_when_artifacts_missing(self):
        client = MagicMock()
        client.list_artifacts.return_value = [types.SimpleNamespace(path='model')]

        with self.assertRaises(RuntimeError) as error:
            train.assert_run_artifacts_present(client, 'production-run')

        self.assertIn('production_bundle', str(error.exception))

    @patch('Scripts.train.mlflow.sklearn.log_model')
    def test_log_model_with_compatible_uri_returns_model_uri(self, log_model):
        log_model.return_value = types.SimpleNamespace(model_uri='models:/m-123')

        model_info, model_uri = train.log_model_with_compatible_uri(object())

        self.assertEqual(model_uri, 'models:/m-123')
        self.assertEqual(model_info.model_uri, 'models:/m-123')
        self.assertEqual(log_model.call_args.kwargs['name'], 'model')

    @patch('Scripts.train.log_bundle_artifact_to_mlflow')
    @patch('Scripts.train.mlflow.register_model')
    @patch('Scripts.train.mlflow.get_run')
    @patch('Scripts.train.log_model_with_compatible_uri')
    @patch('Scripts.train.mlflow.log_metrics')
    @patch('Scripts.train.mlflow.log_param')
    @patch('Scripts.train.mlflow.set_tag')
    @patch('Scripts.train.mlflow.start_run')
    @patch('Scripts.train.mlflow.MlflowClient')
    @patch('Scripts.train.mlflow.set_tracking_uri')
    def test_register_best_model_registers_once(
        self,
        _set_tracking_uri,
        mlflow_client_cls,
        start_run,
        _set_tag,
        _log_param,
        _log_metrics,
        log_model_with_compatible_uri,
        get_run,
        register_model,
        _log_bundle_artifact,
    ):
        run_context = MagicMock()
        run_context.__enter__.return_value = types.SimpleNamespace(info=types.SimpleNamespace(run_id='production-run'))
        run_context.__exit__.return_value = False
        start_run.return_value = run_context

        client = mlflow_client_cls.return_value
        client.search_model_versions.return_value = [types.SimpleNamespace(version='2', current_stage='Production', run_id='old-run')]
        client.list_artifacts.return_value = [types.SimpleNamespace(path='production_bundle')]
        get_run.return_value = types.SimpleNamespace(data=types.SimpleNamespace(metrics={'val_f1': 0.70}))
        log_model_with_compatible_uri.return_value = (types.SimpleNamespace(model_uri='models:/m-123'), 'models:/m-123')
        register_model.return_value = types.SimpleNamespace(version='3')

        best_model_record = {
            'params': {'model_family': 'logistic_regression', 'decision_threshold': 0.52},
            'val_metrics': {'f1': 0.80, 'roc_auc': 0.90, 'log_loss': 0.30},
            'test_metrics': {'f1': 0.79, 'roc_auc': 0.89, 'log_loss': 0.31},
            'business_metrics': {'high_risk_rate': 0.20},
            'model': object(),
            'run_id': 'candidate-run',
            'name': 'lr_candidate',
        }

        train.register_best_model(best_model_record, dv=object(), scaler=object())

        self.assertEqual(register_model.call_count, 1)
        register_model.assert_called_once_with('models:/m-123', train.MODEL_REGISTRY_NAME)


if __name__ == '__main__':
    unittest.main()
