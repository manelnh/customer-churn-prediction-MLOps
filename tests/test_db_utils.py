import unittest
from unittest.mock import patch

from Scripts import db_utils


class FakeCursor:
    def __init__(self, connection):
        self.connection = connection
        self._fetchone_result = [101]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, query, params=None):
        self.connection.executed.append((query, params))

    def fetchone(self):
        return self._fetchone_result

    def fetchall(self):
        return []


class FakeConnection:
    def __init__(self):
        self.executed = []
        self.commit_count = 0
        self.rollback_count = 0

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        self.commit_count += 1

    def rollback(self):
        self.rollback_count += 1


class DbUtilsTests(unittest.TestCase):
    def test_insert_prediction_log_commits_and_returns_id(self):
        connection = FakeConnection()

        prediction_id = db_utils.insert_prediction_log(
            connection,
            input_features={'tenure': 12},
            predicted_probability=0.73,
            predicted_label='Churn',
            predicted_risk='HIGH RISK',
            top_drivers=[{'feature': 'tenure', 'impact': 0.4}],
        )

        self.assertEqual(prediction_id, 101)
        self.assertEqual(connection.commit_count, 1)
        self.assertEqual(connection.rollback_count, 0)
        self.assertIn('INSERT INTO prediction_logs', connection.executed[0][0])

    def test_update_prediction_ground_truth_sets_expected_query(self):
        connection = FakeConnection()

        db_utils.update_prediction_ground_truth(
            connection,
            prediction_id=5,
            actual_label='No churn',
            ground_truth_source='qa_review',
            feedback_notes='Customer renewed plan',
        )

        _, params = connection.executed[0]
        self.assertEqual(params[0], 'No churn')
        self.assertEqual(params[1], 'LOW RISK')
        self.assertEqual(params[2], 'qa_review')
        self.assertEqual(params[3], 'Customer renewed plan')
        self.assertEqual(params[4], 5)
        self.assertEqual(connection.commit_count, 1)

    @patch('Scripts.db_utils.ensure_platform_tables')
    def test_bootstrap_platform_tables_if_enabled_is_opt_in(self, ensure_tables_mock):
        connection = FakeConnection()

        with patch.dict('os.environ', {}, clear=False):
            enabled = db_utils.bootstrap_platform_tables_if_enabled(connection)

        self.assertFalse(enabled)
        ensure_tables_mock.assert_not_called()


if __name__ == '__main__':
    unittest.main()
