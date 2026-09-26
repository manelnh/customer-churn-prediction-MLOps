import unittest

import pandas as pd

import streamlit_app


class DecisionSupportTests(unittest.TestCase):
    def test_manager_action_catalog_uses_balanced_counts_by_risk(self):
        self.assertEqual(len(streamlit_app.get_manager_action_options('HIGH RISK')), 8)
        self.assertEqual(len(streamlit_app.get_manager_action_options('MEDIUM RISK')), 8)
        self.assertEqual(len(streamlit_app.get_manager_action_options('LOW RISK')), 6)

    def test_ranked_recommendations_start_with_primary_action_and_limit_to_three(self):
        row = pd.Series(
            {
                'predicted_risk': 'HIGH RISK',
                'predicted_probability': 0.84,
                'MonthlyCharges': 92.0,
                'Data_Usage_GB': 24.0,
                'tenure': 8,
                'Support_Tickets': 1,
                'App_Logins': 8,
                'Contract': 'month-to-month',
                'OnlineSecurity': 'No',
                'OnlineBackup': 'Yes',
                'InternetService': 'Fiber optic',
                'PaymentMethod': 'Electronic Check',
            }
        )

        recommendations = streamlit_app.build_ranked_manager_recommendations(row, limit=3)

        self.assertEqual(len(recommendations), 3)
        self.assertEqual(recommendations[0]['rank'], 1)
        self.assertEqual(recommendations[0]['action'], 'Retention call with data plan offer')

    def test_default_manager_action_prefers_top_ranked_recommendation_when_unsaved(self):
        default_action = streamlit_app.get_default_manager_action(
            saved_action=None,
            manager_options=['Urgent retention call', 'Retention call with data plan offer', 'Email follow-up'],
            ranked_recommendations=[
                {'rank': 1, 'action': 'Retention call with data plan offer', 'why': 'Best fit.'},
                {'rank': 2, 'action': 'Urgent retention call', 'why': 'Alternative.'},
            ],
            recommended_action='Urgent retention call',
        )

        self.assertEqual(default_action, 'Retention call with data plan offer')

    def test_very_low_risk_customer_falls_back_to_loyalty_style_action(self):
        row = pd.Series(
            {
                'predicted_risk': 'LOW RISK',
                'predicted_probability': 0.05,
                'MonthlyCharges': 95.0,
                'Data_Usage_GB': 24.0,
                'tenure': 18,
                'Support_Tickets': 0,
                'App_Logins': 12,
                'Contract': 'month-to-month',
                'OnlineSecurity': 'Yes',
                'OnlineBackup': 'Yes',
                'InternetService': 'Fiber optic',
                'PaymentMethod': 'Electronic Check',
            }
        )

        playbook = streamlit_app.build_customer_retention_playbook(row)

        self.assertEqual(playbook['primary_category'], 'general')
        self.assertEqual(playbook['recommended_action'], 'Thank-you message')
        self.assertIn('Very low churn probability', playbook['reason_summary'])

    def test_low_risk_customer_stays_on_general_monitoring_path(self):
        row = pd.Series(
            {
                'predicted_risk': 'LOW RISK',
                'predicted_probability': 0.34,
                'MonthlyCharges': 95.0,
                'Data_Usage_GB': 24.0,
                'tenure': 18,
                'Support_Tickets': 1,
                'App_Logins': 8,
                'Contract': 'month-to-month',
                'OnlineSecurity': 'Yes',
                'OnlineBackup': 'Yes',
                'InternetService': 'Fiber optic',
                'PaymentMethod': 'Electronic Check',
            }
        )

        playbook = streamlit_app.build_customer_retention_playbook(row)

        self.assertEqual(playbook['primary_category'], 'general')
        self.assertEqual(playbook['recommended_action'], 'Schedule follow-up')
        self.assertIn('early warning signals', playbook['reason_summary'])

    def test_low_risk_second_decile_uses_check_in_email(self):
        row = pd.Series(
            {
                'predicted_risk': 'LOW RISK',
                'predicted_probability': 0.15,
                'MonthlyCharges': 60.0,
                'Data_Usage_GB': 8.0,
                'tenure': 20,
                'Support_Tickets': 0,
                'App_Logins': 10,
                'Contract': 'one year',
                'OnlineSecurity': 'Yes',
                'OnlineBackup': 'Yes',
                'InternetService': 'DSL',
                'PaymentMethod': 'Credit Card (automatic)',
            }
        )

        playbook = streamlit_app.build_customer_retention_playbook(row)

        self.assertEqual(playbook['recommended_action'], 'Email follow-up')
        self.assertEqual(playbook['probability_band_label'], '11-20%')

    def test_low_risk_engagement_case_uses_tailored_light_reactivation_action(self):
        row = pd.Series(
            {
                'predicted_risk': 'LOW RISK',
                'predicted_probability': 0.18,
                'MonthlyCharges': 58.0,
                'Data_Usage_GB': 7.0,
                'tenure': 18,
                'Support_Tickets': 0,
                'App_Logins': 3,
                'Contract': 'one year',
                'OnlineSecurity': 'Yes',
                'OnlineBackup': 'Yes',
                'InternetService': 'DSL',
                'PaymentMethod': 'Credit Card (automatic)',
            }
        )

        playbook = streamlit_app.build_customer_retention_playbook(row)

        self.assertEqual(playbook['primary_category'], 'engagement')
        self.assertEqual(playbook['recommended_action'], 'Email follow-up')
        self.assertEqual(playbook['campaign'], 'Light Engagement')

    def test_low_risk_contract_case_uses_contract_watch_action(self):
        row = pd.Series(
            {
                'predicted_risk': 'LOW RISK',
                'predicted_probability': 0.19,
                'MonthlyCharges': 61.0,
                'Data_Usage_GB': 9.0,
                'tenure': 24,
                'Support_Tickets': 0,
                'App_Logins': 9,
                'Contract': 'month-to-month',
                'OnlineSecurity': 'Yes',
                'OnlineBackup': 'Yes',
                'InternetService': 'DSL',
                'PaymentMethod': 'Credit Card (automatic)',
            }
        )

        playbook = streamlit_app.build_customer_retention_playbook(row)

        self.assertEqual(playbook['primary_category'], 'contract')
        self.assertEqual(playbook['recommended_action'], 'Watch contract end date')
        self.assertEqual(playbook['campaign'], 'Contract Watch')

    def test_build_customer_retention_playbook_prefers_data_usage_campaign(self):
        row = pd.Series(
            {
                'predicted_risk': 'HIGH RISK',
                'predicted_probability': 0.84,
                'MonthlyCharges': 92.0,
                'Data_Usage_GB': 24.0,
                'tenure': 8,
                'Support_Tickets': 1,
                'App_Logins': 8,
                'Contract': 'month-to-month',
                'OnlineSecurity': 'No',
                'OnlineBackup': 'Yes',
                'InternetService': 'Fiber optic',
                'PaymentMethod': 'Electronic Check',
            }
        )

        playbook = streamlit_app.build_customer_retention_playbook(row)

        self.assertEqual(playbook['primary_category'], 'data_usage')
        self.assertIn('Data Usage', playbook['campaign'])
        self.assertIn('Gold', playbook['offer_detail'])
        self.assertEqual(playbook['recommended_action'], 'Retention call with data plan offer')

    def test_build_customer_retention_playbook_prefers_support_recovery_when_friction_dominates(self):
        row = pd.Series(
            {
                'predicted_risk': 'HIGH RISK',
                'predicted_probability': 0.88,
                'MonthlyCharges': 68.0,
                'Data_Usage_GB': 6.0,
                'tenure': 20,
                'Support_Tickets': 5,
                'App_Logins': 6,
                'Contract': 'one year',
                'OnlineSecurity': 'Yes',
                'OnlineBackup': 'Yes',
                'InternetService': 'DSL',
                'PaymentMethod': 'Credit Card (automatic)',
            }
        )

        playbook = streamlit_app.build_customer_retention_playbook(row)

        self.assertEqual(playbook['primary_category'], 'support')
        self.assertIn('Service Recovery', playbook['campaign'])

    def test_combo_playbook_uses_contract_save_when_engagement_and_contract_risk_align(self):
        row = pd.Series(
            {
                'predicted_risk': 'HIGH RISK',
                'predicted_probability': 0.84,
                'MonthlyCharges': 68.0,
                'Data_Usage_GB': 8.0,
                'tenure': 20,
                'Support_Tickets': 1,
                'App_Logins': 1,
                'Contract': 'month-to-month',
                'OnlineSecurity': 'Yes',
                'OnlineBackup': 'Yes',
                'InternetService': 'DSL',
                'PaymentMethod': 'Credit Card (automatic)',
            }
        )

        playbook = streamlit_app.build_customer_retention_playbook(row)

        self.assertEqual(playbook['primary_category'], 'contract')
        self.assertEqual(playbook['recommended_action'], 'Retention call with plan upgrade')
        self.assertEqual(playbook['campaign'], 'Reactivation + Contract Save')
        self.assertEqual(playbook['secondary_reason'], 'Very low digital engagement suggests detachment')

    def test_combo_playbook_uses_senior_recovery_when_support_and_billing_both_peak(self):
        row = pd.Series(
            {
                'predicted_risk': 'HIGH RISK',
                'predicted_probability': 0.92,
                'MonthlyCharges': 95.0,
                'Data_Usage_GB': 6.0,
                'tenure': 10,
                'Support_Tickets': 5,
                'App_Logins': 8,
                'Contract': 'two year',
                'OnlineSecurity': 'Yes',
                'OnlineBackup': 'Yes',
                'InternetService': 'DSL',
                'PaymentMethod': 'Credit Card (automatic)',
            }
        )

        playbook = streamlit_app.build_customer_retention_playbook(row)

        self.assertEqual(playbook['primary_category'], 'support')
        self.assertEqual(playbook['recommended_action'], 'Service recovery escalation')
        self.assertEqual(playbook['campaign'], 'Service Recovery + Value Save')
        self.assertIn('service pain', playbook['reason_summary'].lower())

    def test_playbook_exposes_customer_specific_top_drivers_for_action_alignment(self):
        row = pd.Series(
            {
                'predicted_risk': 'HIGH RISK',
                'predicted_probability': 0.81,
                'MonthlyCharges': 98.0,
                'Data_Usage_GB': 23.0,
                'tenure': 5,
                'Support_Tickets': 1,
                'App_Logins': 4,
                'Contract': 'month-to-month',
                'OnlineSecurity': 'No',
                'OnlineBackup': 'No',
                'InternetService': 'Fiber optic',
                'PaymentMethod': 'Electronic Check',
                'top_drivers': [
                    {'feature': 'Data_Usage_GB', 'impact': 1.4},
                    {'feature': 'MonthlyCharges', 'impact': 1.1},
                    {'feature': 'Contract=Month-to-month', 'impact': 0.8},
                ],
            }
        )

        playbook = streamlit_app.build_customer_retention_playbook(row)
        guidance = streamlit_app.get_manager_action_guidance(row)

        self.assertEqual(playbook['primary_category'], 'data_usage')
        self.assertIn('Data usage is high', playbook['top_driver_labels'])
        self.assertIn('Monthly charges are high', playbook['top_driver_labels'])
        self.assertEqual(guidance['watchout_title'], 'Why this customer may leave')
        self.assertIn('Top customer-specific drivers', playbook['coaching_note'])

    def test_medium_risk_data_usage_case_uses_silver_pack_band_action(self):
        row = pd.Series(
            {
                'predicted_risk': 'MEDIUM RISK',
                'predicted_probability': 0.66,
                'MonthlyCharges': 82.0,
                'Data_Usage_GB': 21.0,
                'tenure': 14,
                'Support_Tickets': 1,
                'App_Logins': 9,
                'Contract': 'one year',
                'OnlineSecurity': 'No',
                'OnlineBackup': 'Yes',
                'InternetService': 'Fiber optic',
                'PaymentMethod': 'Electronic Check',
            }
        )

        playbook = streamlit_app.build_customer_retention_playbook(row)

        self.assertEqual(playbook['primary_category'], 'data_usage')
        self.assertEqual(playbook['recommended_action'], 'Data plan offer by SMS')
        self.assertEqual(playbook['offer_detail'], 'Silver pack')

    def test_customer_action_brief_uses_detailed_factor_and_driver_sections_for_medium_risk(self):
        row = pd.Series(
            {
                'predicted_risk': 'MEDIUM RISK',
                'predicted_probability': 0.67,
                'MonthlyCharges': 88.0,
                'Data_Usage_GB': 21.0,
                'tenure': 11,
                'Support_Tickets': 2,
                'App_Logins': 4,
                'Contract': 'month-to-month',
                'OnlineSecurity': 'No',
                'OnlineBackup': 'No',
                'InternetService': 'Fiber optic',
                'PaymentMethod': 'Electronic Check',
                'top_drivers': [
                    {'feature': 'Data_Usage_GB', 'impact': 1.3},
                    {'feature': 'MonthlyCharges', 'impact': 1.0},
                    {'feature': 'App_Logins', 'impact': 0.7},
                ],
            }
        )

        brief = streamlit_app.build_customer_action_brief(row)

        self.assertEqual(brief['factor_heading'], 'Factors to catch early')
        self.assertTrue(brief['show_driver_features'])
        self.assertGreaterEqual(len(brief['factor_items']), 1)
        self.assertIn('Data usage is high', brief['top_driver_labels'])

    def test_build_playbook_portfolio_frame_groups_cases_by_campaign(self):
        logs = pd.DataFrame(
            [
                {
                    'predicted_risk': 'HIGH RISK',
                    'predicted_probability': 0.82,
                    'MonthlyCharges': 90.0,
                    'Revenue at Risk': 90.0,
                    'Data_Usage_GB': 22.0,
                    'tenure': 10,
                    'Support_Tickets': 1,
                    'App_Logins': 9,
                    'Contract': 'month-to-month',
                    'OnlineSecurity': 'No',
                    'OnlineBackup': 'Yes',
                    'InternetService': 'Fiber optic',
                    'PaymentMethod': 'Electronic Check',
                },
                {
                    'predicted_risk': 'MEDIUM RISK',
                    'predicted_probability': 0.63,
                    'MonthlyCharges': 78.0,
                    'Revenue at Risk': 39.0,
                    'Data_Usage_GB': 16.0,
                    'tenure': 14,
                    'Support_Tickets': 1,
                    'App_Logins': 10,
                    'Contract': 'month-to-month',
                    'OnlineSecurity': 'No',
                    'OnlineBackup': 'Yes',
                    'InternetService': 'Fiber optic',
                    'PaymentMethod': 'Electronic Check',
                },
            ]
        )

        portfolio = streamlit_app.build_playbook_portfolio_frame(logs)

        self.assertEqual(int(portfolio['customers'].sum()), 2)
        self.assertIn('data_usage', portfolio['category'].tolist())
        self.assertGreaterEqual(len(portfolio), 1)


if __name__ == '__main__':
    unittest.main()
