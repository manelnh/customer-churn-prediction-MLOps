"""
TelCo Churn Prediction - Managerial Decision Support System
===============================================================
A Streamlit MLOps dashboard for customer churn prediction with executive insights,
actionable recommendations, and technical model governance.
"""

import os
import sys
import json
import base64
import logging
import subprocess
from urllib import error as urllib_error
from urllib import request as urllib_request
import warnings
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import Scripts.db_utils as db_utils

# Suppress Git/Python warnings for cleaner output
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
logging.getLogger("mlflow").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Import local modules
from Scripts.db_utils import (
    get_postgres_connection,
    bootstrap_platform_tables_if_enabled,
    insert_prediction_log,
    insert_governance_decision,
    update_prediction_ground_truth,
    replace_monitoring_alerts,
    fetch_all_predictions as load_prediction_logs,
    fetch_recent_alerts as load_alerts,
    fetch_recent_governance_decisions as load_governance_decisions,
)
from Scripts.model_utils import (
    DEFAULT_DECISION_THRESHOLD,
    calculate_model_selection_score,
    classify_from_probability,
    filter_predictions_for_active_model,
    get_active_bundle_metadata,
    get_baseline_metrics_from_metadata,
    load_model_bundle,
    load_bundle,
    prepare_customer_features,
    get_risk_label,
    calculate_live_metrics,
    detect_monitoring_alerts,
    build_probability_drift_frame,
    build_lifecycle_frame,
    build_retraining_recommendation,
    calculate_model_trust_score,
    serialize_top_drivers,
    explain_top_drivers,
    run_lab_experiment_cached,
    log_lab_run_to_mlflow,
    build_lab_run_name,
    get_mlflow_tracking_uri,
    get_mlflow_experiment_name,
)
from Scripts.model_utils import get_production_baseline_metrics, get_production_baseline_params

# --- Configuration Constants ---
ROOT_DIR = Path(__file__).parent.resolve()
PAGE_ICON = '📊'
PRODUCTION_BASELINE_METRICS = get_production_baseline_metrics()
PRODUCTION_BASELINE_PARAMS = get_production_baseline_params()
FORM_SOURCE = 'streamlit_ui'
DEFAULT_RISK_ACTION_CONFIG = {
    'HIGH RISK': 'Phone Call - Priority 1',
    'MEDIUM RISK': 'SMS Discount Offer',
    'LOW RISK': 'No Action Needed',
}
ACTION_DISPLAY_LABELS = {
    'Phone Call - Priority 1': 'Urgent retention call',
    'Immediate Retention Call + 15% Discount Offer': 'Urgent retention call',
    'Retention Call + 10% Discount Offer': 'Retention call with discount',
    'Escalate to Account Manager': 'Account manager escalation',
    'Retention Call + Gold Data Pack Offer': 'Retention call with data plan offer',
    'Account Manager Call + Gold Data Pack Offer': 'Retention call with data plan offer',
    'Escalate to Service Recovery Team': 'Service recovery escalation',
    'Account Manager + Service Recovery Escalation': 'Service recovery escalation',
    'Retention Call + Contract Upgrade Offer': 'Retention call with plan upgrade',
    'Retention Call + Premium Contract Upgrade Offer': 'Retention call with plan upgrade',
    'Account Manager Call + Premium Contract Upgrade Offer': 'Retention call with plan upgrade',
    'Retention Call + Welcome Benefit': 'Retention call with onboarding offer',
    'Retention Call + Premium Welcome Benefit': 'Retention call with onboarding offer',
    'Account Manager Call + Premium Welcome Benefit': 'Retention call with onboarding offer',
    'Retention Call + Security/Backup Bundle': 'Add-on package offer',
    'Retention Call + Premium Bundle Offer': 'Add-on package offer',
    'Account Manager Call + Premium Bundle Offer': 'Add-on package offer',
    'Retention Call + Usage Coaching': 'Retention call with usage coaching',
    'Retention Call + Reactivation Offer': 'Retention call with win-back offer',
    'Account Manager Call + Reactivation Offer': 'Retention call with win-back offer',
    'Account Manager Call + 15% Discount Offer': 'Urgent retention call',
    'SMS Discount Offer': 'Discount offer by SMS',
    'Nurture Campaign + Check-in Call': 'Check-in call',
    'Personalized Email Offer': 'Email follow-up',
    'Schedule Follow-up Review': 'Schedule follow-up',
    'Personalized Check-In Email': 'Email follow-up',
    'SMS Offer - Silver Data Pack': 'Data plan offer by SMS',
    'SMS Offer - Bronze Data Pack': 'Data plan offer by SMS',
    'Priority Support Follow-up': 'Support follow-up',
    'Service Check-In SMS': 'Service follow-up SMS',
    'Retention Call + Service Recovery': 'Retention call with service recovery',
    'Personalized Contract Upgrade Offer': 'Plan upgrade offer',
    'Renewal Reminder + Value Email': 'Email follow-up',
    'Personalized Bill Relief Offer': 'Email follow-up',
    'Personalized Savings Email': 'Email follow-up',
    'Personalized Add-on Bundle Offer': 'Add-on package offer',
    'Add-on Discovery Email': 'Email follow-up',
    'Onboarding Check-in Journey': 'Check-in call',
    'Welcome Check-In Email': 'Email follow-up',
    'Usage Tips Email': 'Email follow-up',
    'No Action Needed': 'No action for now',
    'Monitor Only': 'Keep monitoring',
    'Loyalty Thank-You Message': 'Thank-you message',
    'Monitor Data-Pack Eligibility': 'Watch for better data plan fit',
    'Monitor Renewal Window': 'Watch contract end date',
    'Monitor Bundle Adoption': 'Watch add-on usage',
}
DEFAULT_ACTION_PLAYBOOKS = {
    'data_usage': {
        'label': 'Usage Pressure',
        'campaign': 'Data Usage Retention',
        'recommended_action': {
            'HIGH RISK': 'Retention Call + Gold Data Pack Offer',
            'MEDIUM RISK': 'SMS Offer - Silver Data Pack',
            'LOW RISK': 'Monitor Data-Pack Eligibility',
        },
        'next_step': 'Offer a lower-cost data bundle sized to the customer usage pattern.',
    },
    'billing': {
        'label': 'Price Sensitivity',
        'campaign': 'Price Protection',
        'recommended_action': {
            'HIGH RISK': 'Immediate Retention Call + 15% Discount Offer',
            'MEDIUM RISK': 'Personalized Bill Relief Offer',
            'LOW RISK': 'Monitor Only',
        },
        'next_step': 'Reduce price pressure with a targeted discount or bill optimization offer.',
    },
    'support': {
        'label': 'Service Friction',
        'campaign': 'Service Recovery',
        'recommended_action': {
            'HIGH RISK': 'Escalate to Service Recovery Team',
            'MEDIUM RISK': 'Priority Support Follow-up',
            'LOW RISK': 'Monitor Only',
        },
        'next_step': 'Resolve service issues first, then follow with a retention touchpoint.',
    },
    'engagement': {
        'label': 'Low Engagement',
        'campaign': 'Customer Reactivation',
        'recommended_action': {
            'HIGH RISK': 'Retention Call + Usage Coaching',
            'MEDIUM RISK': 'Nurture Campaign + Check-in Call',
            'LOW RISK': 'Loyalty Thank-You Message',
        },
        'next_step': 'Rebuild product attachment with onboarding, tutorials, and digital nudges.',
    },
    'contract': {
        'label': 'Contract Flexibility',
        'campaign': 'Contract Conversion',
        'recommended_action': {
            'HIGH RISK': 'Retention Call + Contract Upgrade Offer',
            'MEDIUM RISK': 'Personalized Contract Upgrade Offer',
            'LOW RISK': 'Monitor Renewal Window',
        },
        'next_step': 'Move customers toward a stickier plan with better long-term value.',
    },
    'tenure': {
        'label': 'Early Lifecycle',
        'campaign': 'Early Lifecycle Retention',
        'recommended_action': {
            'HIGH RISK': 'Retention Call + Welcome Benefit',
            'MEDIUM RISK': 'Onboarding Check-in Journey',
            'LOW RISK': 'Loyalty Thank-You Message',
        },
        'next_step': 'Reassure newer customers before habits and loyalty are fully formed.',
    },
    'bundle': {
        'label': 'Weak Bundle Attachment',
        'campaign': 'Bundle Attachment',
        'recommended_action': {
            'HIGH RISK': 'Retention Call + Security/Backup Bundle',
            'MEDIUM RISK': 'Personalized Add-on Bundle Offer',
            'LOW RISK': 'Monitor Bundle Adoption',
        },
        'next_step': 'Increase switching cost by adding sticky services the customer will value.',
    },
    'general': {
        'label': 'General Retention',
        'campaign': 'Standard Retention',
        'recommended_action': {
            'HIGH RISK': 'Phone Call - Priority 1',
            'MEDIUM RISK': 'SMS Discount Offer',
            'LOW RISK': 'No Action Needed',
        },
        'next_step': 'Use the default retention path until a stronger churn cause emerges.',
    },
}
TUNISIA_GOVERNORATES = [
    'Ariana',
    'Beja',
    'Ben Arous',
    'Bizerte',
    'Gabes',
    'Gafsa',
    'Jendouba',
    'Kairouan',
    'Kasserine',
    'Kebili',
    'Kef',
    'Mahdia',
    'Manouba',
    'Medenine',
    'Monastir',
    'Nabeul',
    'Sfax',
    'Sidi Bouzid',
    'Siliana',
    'Sousse',
    'Tataouine',
    'Tozeur',
    'Tunis',
    'Zaghouan',
]


def get_bundle_metadata() -> dict:
    metadata = get_active_bundle_metadata()
    if metadata:
        return metadata

    candidate_paths = [ROOT_DIR / 'churn_production.pkl', ROOT_DIR / 'models' / 'churn_production_bundle.pkl']
    for path in candidate_paths:
        if path.exists():
            try:
                return load_bundle(path).get('metadata', {})
            except Exception:
                continue
    return {}


def load_action_logic_config() -> dict:
    config_path = ROOT_DIR / 'next_best_action_config.json'
    config = {
        'risk_actions': DEFAULT_RISK_ACTION_CONFIG.copy(),
        'playbooks': DEFAULT_ACTION_PLAYBOOKS.copy(),
    }
    if not config_path.exists():
        return config

    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            raw_config = json.load(file)
    except Exception:
        return config

    if all(key in raw_config for key in DEFAULT_RISK_ACTION_CONFIG):
        config['risk_actions'].update({key: raw_config.get(key) for key in DEFAULT_RISK_ACTION_CONFIG})
        return config

    risk_actions = raw_config.get('risk_actions', {})
    if isinstance(risk_actions, dict):
        for key in DEFAULT_RISK_ACTION_CONFIG:
            config['risk_actions'][key] = risk_actions.get(key, config['risk_actions'][key])

    playbooks = raw_config.get('playbooks', {})
    if isinstance(playbooks, dict):
        for category, details in playbooks.items():
            if category not in config['playbooks'] or not isinstance(details, dict):
                continue
            merged = config['playbooks'][category].copy()
            merged.update(details)
            if isinstance(details.get('recommended_action'), dict):
                action_map = merged['recommended_action'].copy()
                action_map.update(details['recommended_action'])
                merged['recommended_action'] = action_map
            config['playbooks'][category] = merged

    return config


def save_action_logic_config(config: dict):
    config_path = ROOT_DIR / 'next_best_action_config.json'
    with open(config_path, 'w', encoding='utf-8') as file:
        json.dump(config, file, indent=2)


def normalize_action_label(action_name: str) -> str:
    return ACTION_DISPLAY_LABELS.get(str(action_name or '').strip(), str(action_name or '').strip())


def classify_action_tone(action_name: str) -> str:
    normalized_action = normalize_action_label(action_name).lower()
    if not normalized_action or normalized_action == 'not selected':
        return 'neutral'
    if any(token in normalized_action for token in ['urgent', 'escalate', 'account manager-led', 'service recovery team']):
        return 'critical'
    if 'retention call' in normalized_action or 'account manager call' in normalized_action:
        return 'warning'
    if any(token in normalized_action for token in ['sms', 'email', 'check-in', 'follow-up', 'onboarding', 'renewal reminder']):
        return 'informational'
    if any(token in normalized_action for token in ['no action', 'keep monitoring', 'thank-you', 'watch ']):
        return 'healthy'
    return 'neutral'


def render_section_anchor(anchor_id: str):
    st.markdown(f'<div id="{anchor_id}" class="page-anchor"></div>', unsafe_allow_html=True)


def _render_sidebar_stat_card(
    label: str,
    value: str,
    tone: str = 'neutral',
    detail: str = '',
    centered: bool = False,
    compact: bool = False,
):
    detail_html = f'<div class="sidebar-stat-detail">{detail}</div>' if detail else ''
    alignment_class = ' sidebar-stat-card-centered' if centered else ''
    compact_class = ' sidebar-stat-card-compact' if compact else ''
    st.markdown(
        (
            f'<div class="sidebar-stat-card sidebar-stat-{tone}{alignment_class}{compact_class}">'
            f'<div class="sidebar-stat-label">{label}</div>'
            f'<div class="sidebar-stat-value">{value}</div>'
            f'{detail_html}'
            '</div>'
        ),
        unsafe_allow_html=True,
    )


def _render_sidebar_guide_item(label: str, detail: str):
    st.markdown(
        (
            '<div class="sidebar-guide-item">'
            f'<div class="sidebar-guide-label">{label}</div>'
            f'<div class="sidebar-guide-detail">{detail}</div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )


def calculate_revenue_at_risk_total(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    if 'Revenue at Risk' in df.columns:
        return float(pd.to_numeric(df['Revenue at Risk'], errors='coerce').fillna(0.0).sum())
    monthly_charges = pd.to_numeric(df.get('MonthlyCharges'), errors='coerce').fillna(0.0)
    predicted_risk = df.get('predicted_risk', pd.Series(index=df.index, dtype='object'))
    return float(np.where(predicted_risk.eq('HIGH RISK'), monthly_charges, 0.0).sum())


def render_sidebar_decision_coach(predictions: pd.DataFrame, alerts_df: pd.DataFrame, governance_df: pd.DataFrame):
    decision_df = build_decision_support_frame(predictions)
    high_risk_count = int((decision_df.get('predicted_risk') == 'HIGH RISK').sum()) if not decision_df.empty else 0
    medium_risk_count = int((decision_df.get('predicted_risk') == 'MEDIUM RISK').sum()) if not decision_df.empty else 0
    low_risk_count = int((decision_df.get('predicted_risk') == 'LOW RISK').sum()) if not decision_df.empty else 0
    pending_action_count = int((decision_df.get('Action Status') == 'Pending Action').sum()) if not decision_df.empty else 0
    pending_high_risk_count = int(
        ((decision_df.get('Action Status') == 'Pending Action') & (decision_df.get('predicted_risk') == 'HIGH RISK')).sum()
    ) if not decision_df.empty else 0
    revenue_at_risk = calculate_revenue_at_risk_total(decision_df)

    alert_records = alerts_df.to_dict('records') if not alerts_df.empty else []
    high_severity_alerts = sum(1 for alert in alert_records if str(alert.get('severity', '')).lower() == 'high')
    trust_score = calculate_model_trust_score(alert_records) if alert_records else 100
    latest_governance = governance_df.iloc[0].to_dict() if not governance_df.empty else {}
    latest_governance_decision = latest_governance.get('decision', 'No decision yet')

    if high_severity_alerts > 0:
        alert_label = 'alert' if high_severity_alerts == 1 else 'alerts'
        content = {
            'title': 'Monitoring Checkpoint',
            'subtitle': 'Review model health before expanding action volume.',
            'body': (
                f'{high_severity_alerts} high-severity {alert_label} are currently active. '
                'Use Monitoring & Retraining to confirm decision quality before pushing broader retention outreach.'
            ),
        }
    elif pending_high_risk_count > 0:
        content = {
            'title': 'Execution Bottleneck',
            'subtitle': 'High-risk cases are waiting on action.',
            'body': f'{pending_high_risk_count} high-risk customer(s) still need a manager action, so the operational priority is to clear those cases first.',
        }
    elif decision_df.empty:
        content = {
            'title': 'Ready To Launch',
            'subtitle': 'No active customer queue yet.',
            'body': 'No saved customer cases are available yet. Score a customer to begin the decision and monitoring flow.',
        }
    else:
        content = {
            'title': 'Portfolio In Motion',
            'subtitle': 'Risk is active and decisions are flowing.',
            'body': f'{high_risk_count} high-risk and {medium_risk_count} medium-risk cases are active. Use the workspace to prioritize intervention and track execution quality.',
        }

    st.markdown(
        (
            '<div class="sidebar-panel sidebar-panel-hero">'
            '<div class="sidebar-panel-kicker">Decision Command</div>'
            '<div class="sidebar-coach-card">'
            f'<div class="sidebar-coach-kicker">{content["title"]}</div>'
            f'<div class="sidebar-coach-title">{content["subtitle"]}</div>'
            f'<div class="sidebar-coach-body">{content["body"]}</div>'
            '</div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sidebar-section-title">Executive Snapshot</div>', unsafe_allow_html=True)
    _render_sidebar_stat_card(
        'Revenue At Risk',
        format_currency_tnd(revenue_at_risk),
        tone='revenue',
        detail='Current exposed monthly value',
        centered=True,
        compact=True,
    )

    risk_cols = st.columns(3)
    with risk_cols[0]:
        _render_sidebar_stat_card('High Risk', str(high_risk_count), tone='critical')
    with risk_cols[1]:
        _render_sidebar_stat_card('Medium Risk', str(medium_risk_count), tone='warning')
    with risk_cols[2]:
        _render_sidebar_stat_card('Low Risk', str(low_risk_count), tone='healthy')

    st.markdown('<div class="sidebar-section-title">Operational Queue</div>', unsafe_allow_html=True)
    queue_cols = st.columns(2)
    with queue_cols[0]:
        _render_sidebar_stat_card('Pending Actions', str(pending_action_count), tone='neutral', detail='Cases waiting for owner action')
    with queue_cols[1]:
        _render_sidebar_stat_card('Urgent Queue', str(pending_high_risk_count), tone='critical' if pending_high_risk_count > 0 else 'healthy', detail='High-risk cases waiting for action')

    st.markdown('<div class="sidebar-section-title">Recommended Route</div>', unsafe_allow_html=True)
    if high_severity_alerts > 0:
        route_text = 'Open `Monitoring & Retraining` first and verify whether drift or performance change is weakening decision quality.'
    elif pending_high_risk_count > 0:
        route_text = 'Open `Action Center` first and clear the unresolved high-risk queue before reviewing lower-priority cases.'
    elif decision_df.empty:
        route_text = 'Open `Predictions` first and score the next customer case to start building the operating queue.'
    else:
        route_text = 'Open `Manager Insights` first to review where churn pressure, revenue exposure, and campaign demand are concentrated.'
    st.markdown(
        f'<div class="sidebar-route-card"><div class="sidebar-route-label">Next best workspace</div><div class="sidebar-route-body">{route_text}</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sidebar-section-title">Workspace Guide</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-guide-panel">', unsafe_allow_html=True)
    _render_sidebar_guide_item('Manager Insights', 'Portfolio view for risk concentration and business exposure')
    _render_sidebar_guide_item('Predictions', 'Customer-level scoring, explanation, and logging')
    _render_sidebar_guide_item('Action Center', 'Action assignment, queue handling, and follow-through tracking')
    _render_sidebar_guide_item('Monitoring & Retraining', 'Production health, alerts, and retraining triggers')
    _render_sidebar_guide_item('Deep-Dive Analytics', 'Segment-level analysis and churn patterns')
    _render_sidebar_guide_item('Governance & Lab', 'Candidate testing and approval decisions')
    st.markdown('</div>', unsafe_allow_html=True)

# --- Custom Styling ---
def inject_custom_style():
    st.markdown(
        """
        <style>
        .main-header {
            font-size: 2.35rem;
            font-weight: 800;
            color: #0f3057;
            margin: 0;
            letter-spacing: -0.03em;
        }
        .sub-header {
            font-size: 1.4rem;
            font-weight: 600;
            color: #1d3557;
        }
        .hero-title-wrap {
            display: flex;
            align-items: center;
            gap: 14px;
            margin-bottom: 0.75rem;
            padding: 0.95rem 1.15rem;
            background: linear-gradient(135deg, rgba(240, 247, 255, 0.95) 0%, rgba(226, 239, 255, 0.9) 100%);
            border: 1px solid rgba(15, 98, 168, 0.14);
            border-radius: 16px;
            box-shadow: 0 10px 28px rgba(15, 48, 87, 0.08);
        }
        .hero-title-icon {
            width: 56px;
            height: 56px;
            border-radius: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.9rem;
            background: linear-gradient(135deg, #0f62a8 0%, #3aa0d8 100%);
            color: white;
            flex-shrink: 0;
            box-shadow: 0 10px 20px rgba(15, 98, 168, 0.22);
        }
        .hero-title-copy {
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }
        .hero-title-subtitle {
            color: #5f6f82;
            font-size: 1rem;
            font-weight: 500;
            letter-spacing: 0.01em;
        }
        .metric-card {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 12px;
            padding: 1.2rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #f8f9fa;
            border-radius: 8px 8px 0px 0px;
            padding: 10px 20px;
            font-weight: 600;
            transition: all 0.2s ease;
        }
        .stTabs [aria-selected="true"] {
            background-color: #0f62a8;
            color: white;
        }
        section[data-testid="stSidebar"] {
            min-width: 22rem !important;
            max-width: 22rem !important;
            background:
                radial-gradient(circle at top right, rgba(58, 160, 216, 0.12), transparent 32%),
                linear-gradient(180deg, #f8fbff 0%, #f3f7fb 100%);
        }
        section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] {
            display: none !important;
        }
        [data-testid="collapsedControl"] {
            display: none !important;
        }
        section[data-testid="stSidebar"] .block-container {
            padding-top: 1.1rem;
            padding-bottom: 1rem;
        }
        .manager-action-banner {
            padding: 1rem;
            border-radius: 10px;
            font-weight: 700;
            text-align: center;
            margin: 0.5rem 0;
        }
        .manager-action-banner.critical {
            background: linear-gradient(135deg, #fde2e4 0%, #f8d7da 100%);
            color: #9d0208;
            border: 2px solid #9d0208;
        }
        .manager-action-banner.warning {
            background: linear-gradient(135deg, #fff3cd 0%, #ffeeba 100%);
            color: #8d6e00;
            border: 2px solid #8d6e00;
        }
        .manager-action-banner.healthy {
            background: linear-gradient(135deg, #d8f3dc 0%, #b7e4c7 100%);
            color: #1b4332;
            border: 2px solid #1b4332;
        }
        .driver-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin: 0.5rem 0 0.8rem 0;
        }
        .page-anchor {
            position: relative;
            top: -72px;
            visibility: hidden;
        }
        .sidebar-coach-card {
            padding: 1.05rem 1.05rem 1.1rem 1.05rem;
            border-radius: 18px;
            background: linear-gradient(160deg, rgba(255, 255, 255, 0.96) 0%, rgba(239, 246, 252, 0.96) 100%);
            border: 1px solid rgba(18, 53, 91, 0.1);
            box-shadow: 0 10px 22px rgba(18, 53, 91, 0.08);
            margin-bottom: 0.15rem;
        }
        .sidebar-coach-kicker {
            font-size: 0.74rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #6b7c8f;
            margin-bottom: 0.3rem;
        }
        .sidebar-coach-title {
            font-size: 1rem;
            font-weight: 700;
            color: #12355b;
            margin-bottom: 0.25rem;
        }
        .sidebar-coach-body {
            font-size: 0.88rem;
            color: #5f6f82;
            line-height: 1.45;
            margin-bottom: 0.65rem;
        }
        .sidebar-coach-list {
            margin: 0 0 0.8rem 1rem;
            padding: 0;
            color: #355070;
            font-size: 0.88rem;
            line-height: 1.45;
        }
        .sidebar-coach-list li {
            margin-bottom: 0.45rem;
        }
        .sidebar-coach-next-label {
            font-size: 0.74rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #6b7c8f;
            margin-bottom: 0.22rem;
        }
        .sidebar-coach-next {
            padding: 0.72rem 0.78rem;
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.96);
            border: 1px solid rgba(18, 53, 91, 0.08);
            color: #12355b;
            font-size: 0.88rem;
            line-height: 1.42;
            font-weight: 600;
        }
        .driver-chip {
            display: inline-flex;
            align-items: center;
            padding: 0.4rem 0.75rem;
            border-radius: 999px;
            background: linear-gradient(135deg, rgba(18, 53, 91, 0.08) 0%, rgba(47, 143, 157, 0.15) 100%);
            border: 1px solid rgba(47, 143, 157, 0.2);
            color: #12355b;
            font-size: 0.82rem;
            font-weight: 600;
        }
        .sidebar-panel {
            margin-bottom: 1rem;
        }
        .sidebar-panel-hero {
            position: relative;
        }
        .sidebar-panel-kicker {
            font-size: 0.74rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #4d6b88;
            margin-bottom: 0.45rem;
            padding-left: 0.2rem;
        }
        .sidebar-section-title {
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #58718d;
            margin: 1rem 0 0.55rem 0.15rem;
        }
        .sidebar-stat-card {
            min-height: 108px;
            padding: 0.88rem 0.9rem;
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.96);
            border: 1px solid rgba(18, 53, 91, 0.08);
            box-shadow: 0 8px 20px rgba(18, 53, 91, 0.06);
            margin-bottom: 0.55rem;
        }
        .sidebar-stat-card-centered {
            text-align: center;
        }
        .sidebar-stat-card-compact {
            width: 76%;
            margin-left: auto;
            margin-right: auto;
        }
        .sidebar-stat-label {
            font-size: 0.74rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #6a7f95;
            margin-bottom: 0.45rem;
        }
        .sidebar-stat-value {
            font-size: 1.2rem;
            font-weight: 800;
            color: #12355b;
            line-height: 1.1;
            margin-bottom: 0.35rem;
        }
        .sidebar-stat-detail {
            min-height: 2.2rem;
            font-size: 0.79rem;
            line-height: 1.38;
            color: #66788a;
        }
        .sidebar-stat-critical {
            background: linear-gradient(180deg, #fff7f7 0%, #ffe7ea 100%);
            border-color: rgba(181, 54, 84, 0.14);
        }
        .sidebar-stat-warning {
            background: linear-gradient(180deg, #fffaf0 0%, #fff1cc 100%);
            border-color: rgba(201, 150, 34, 0.18);
        }
        .sidebar-stat-healthy {
            background: linear-gradient(180deg, #f3fbf6 0%, #dff3e5 100%);
            border-color: rgba(38, 125, 86, 0.16);
        }
        .sidebar-stat-revenue {
            background: linear-gradient(180deg, #eef7ff 0%, #dfeeff 100%);
            border-color: rgba(15, 98, 168, 0.16);
        }
        .sidebar-route-card {
            padding: 0.9rem 0.95rem;
            border-radius: 16px;
            background: linear-gradient(160deg, #12355b 0%, #1f5d8c 100%);
            box-shadow: 0 12px 24px rgba(18, 53, 91, 0.18);
            margin-bottom: 0.45rem;
        }
        .sidebar-route-label {
            font-size: 0.73rem;
            font-weight: 800;
            letter-spacing: 0.09em;
            text-transform: uppercase;
            color: rgba(255, 255, 255, 0.72);
            margin-bottom: 0.35rem;
        }
        .sidebar-route-body {
            font-size: 0.9rem;
            line-height: 1.45;
            color: #ffffff;
            font-weight: 600;
        }
        .sidebar-guide-panel {
            padding: 0.35rem 0 0.1rem 0;
        }
        .sidebar-guide-item {
            position: relative;
            padding: 0.8rem 0.85rem 0.82rem 1rem;
            margin-bottom: 0.55rem;
            border-radius: 14px;
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid rgba(18, 53, 91, 0.08);
            box-shadow: 0 6px 16px rgba(18, 53, 91, 0.05);
            overflow: hidden;
        }
        .sidebar-guide-item::before {
            content: "";
            position: absolute;
            left: 0;
            top: 0;
            bottom: 0;
            width: 4px;
            background: linear-gradient(180deg, #0f62a8 0%, #43a6dd 100%);
        }
        .sidebar-guide-label {
            font-size: 0.85rem;
            font-weight: 800;
            color: #12355b;
            margin-bottom: 0.18rem;
        }
        .sidebar-guide-detail {
            font-size: 0.8rem;
            line-height: 1.38;
            color: #5f7184;
        }
        .insight-box {
            background: #f0f7ff;
            border-left: 4px solid #0f62a8;
            padding: 1rem;
            border-radius: 0 8px 8px 0;
            margin: 0.5rem 0;
        }
        .decision-badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin: 0.35rem 0 0.85rem 0;
        }
        .decision-badge {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 0.38rem 0.8rem;
            font-size: 0.88rem;
            font-weight: 700;
            border: 1px solid transparent;
        }
        .decision-badge.value-high {
            background: #d8f3dc;
            color: #1b4332;
            border-color: #95d5b2;
        }
        .decision-badge.value-medium {
            background: #e8f4f8;
            color: #0f62a8;
            border-color: #9fd3e8;
        }
        .decision-badge.value-standard {
            background: #f1f3f5;
            color: #495057;
            border-color: #dee2e6;
        }
        .decision-badge.severity-high {
            background: #fde2e4;
            color: #9d0208;
            border-color: #f5b5bb;
        }
        .decision-badge.severity-medium {
            background: #fff3cd;
            color: #8d6e00;
            border-color: #ffe08a;
        }
        .decision-badge.severity-lower {
            background: #e8f7ee;
            color: #2d6a4f;
            border-color: #b7e4c7;
        }
        .decision-badge.action-call {
            background: #dceefb;
            color: #0b4f7a;
            border-color: #9fd3e8;
        }
        .decision-badge.action-offer {
            background: #fff3cd;
            color: #8d6e00;
            border-color: #ffe08a;
        }
        .decision-badge.action-escalate {
            background: #fde2e4;
            color: #9d0208;
            border-color: #f5b5bb;
        }
        .decision-note {
            margin: 0.45rem 0 0.85rem 0;
            padding: 0.7rem 0.9rem;
            border-radius: 12px;
            background: #f7efe2;
            color: #6b4f2a;
            border: 1px solid #ead7b7;
            font-size: 0.92rem;
        }
        .reason-summary-card {
            margin: 0.3rem 0 0.8rem 0;
            padding: 0.95rem 1rem;
            border-radius: 14px;
            background: linear-gradient(135deg, #f7fbff 0%, #edf6ff 100%);
            border: 1px solid #cfe5f6;
            box-shadow: 0 8px 18px rgba(15, 98, 168, 0.08);
        }
        .reason-summary-label {
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: #0f62a8;
            margin-bottom: 0.3rem;
        }
        .reason-summary-text {
            font-size: 1rem;
            font-weight: 600;
            color: #1d3557;
            line-height: 1.45;
        }
        .reason-summary-driver-label {
            display: block;
            margin-top: 0.7rem;
            margin-bottom: 0.15rem;
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: #0f62a8;
        }
        .reason-card {
            margin: 0 0 0.7rem 0;
            padding: 0.85rem 0.95rem;
            border-radius: 14px;
            background: #ffffff;
            border: 1px solid #e4edf5;
            box-shadow: 0 6px 16px rgba(29, 53, 87, 0.06);
        }
        .reason-card-title {
            font-size: 0.95rem;
            font-weight: 700;
            color: #0f3057;
            margin-bottom: 0.25rem;
        }
        .reason-card-body {
            font-size: 0.92rem;
            color: #4f5d75;
            line-height: 1.45;
        }
        .context-summary-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.8rem;
            margin: 0.35rem 0 0.9rem 0;
        }
        .context-summary-card {
            min-width: 160px;
            padding: 0.85rem 0.95rem;
            border-radius: 14px;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            box-shadow: 0 4px 12px rgba(29, 53, 87, 0.05);
        }
        .context-summary-label {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            font-weight: 700;
            color: #6c7a89;
            margin-bottom: 0.28rem;
        }
        .context-summary-value {
            font-size: 1.02rem;
            font-weight: 700;
            color: #0f3057;
        }
        .ranked-rec-grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 0.55rem;
            margin: 0.65rem 0 0.85rem 0;
        }
        .ranked-rec-card {
            padding: 0.82rem 0.9rem 0.84rem 0.9rem;
            border-radius: 14px;
            background: linear-gradient(160deg, #ffffff 0%, #f7fbff 100%);
            border: 1px solid #d9e8f5;
            box-shadow: 0 6px 16px rgba(15, 98, 168, 0.06);
        }
        .ranked-rec-card.is-primary {
            border-color: #f5d27a;
            background: linear-gradient(160deg, #fffdf6 0%, #fff6db 100%);
            box-shadow: 0 10px 22px rgba(191, 127, 0, 0.12);
        }
        .ranked-rec-rank-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.5rem;
            margin-bottom: 0.18rem;
        }
        .ranked-rec-rank {
            font-size: 0.74rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #0f62a8;
        }
        .ranked-rec-badge {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 0.2rem 0.58rem;
            font-size: 0.68rem;
            font-weight: 800;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: #9a6700;
            background: #fff3cd;
            border: 1px solid #f5d27a;
        }
        .ranked-rec-title {
            font-size: 0.92rem;
            font-weight: 800;
            color: #12355b;
            margin-bottom: 0.2rem;
        }
        .ranked-rec-body {
            font-size: 0.8rem;
            color: #5f7184;
            line-height: 1.38;
        }
        .selection-default-note {
            margin: 0.15rem 0 0.55rem 0;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            color: #6b4f2a;
            font-size: 0.84rem;
            font-weight: 600;
        }
        .selection-default-badge {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 0.18rem 0.52rem;
            font-size: 0.67rem;
            font-weight: 800;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: #9a6700;
            background: #fff3cd;
            border: 1px solid #f5d27a;
        }
        .playbook-card {
            margin: 0.35rem 0 0.85rem 0;
            padding: 0.95rem 1rem;
            border-radius: 14px;
            background: linear-gradient(135deg, #fffaf0 0%, #fef3c7 100%);
            border: 1px solid #f5d27a;
            box-shadow: 0 8px 18px rgba(191, 127, 0, 0.08);
        }
        .playbook-kicker {
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: #9a6700;
            margin-bottom: 0.3rem;
        }
        .playbook-title {
            font-size: 1.02rem;
            font-weight: 700;
            color: #6b4f2a;
            margin-bottom: 0.35rem;
        }
        .playbook-body {
            font-size: 0.92rem;
            color: #6b4f2a;
            line-height: 1.5;
        }
        .kpi-container {
            background: white;
            border-radius: 12px;
            padding: 1rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.06);
            text-align: center;
        }
        .kpi-value {
            font-size: 2rem;
            font-weight: 700;
            color: #0f62a8;
        }
        .kpi-label {
            font-size: 0.85rem;
            color: #6c757d;
            margin-top: 0.25rem;
        }
        .status-healthy { color: #2a9d8f; font-weight: 700; }
        .status-warning { color: #f4a261; font-weight: 700; }
        .status-critical { color: #d62839; font-weight: 700; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# --- Data Loading Functions ---
@st.cache_data
def load_dataset() -> pd.DataFrame:
    dataset_path = ROOT_DIR / 'telco_churn_cleaned.csv'
    return pd.read_csv(dataset_path)


@st.cache_resource(show_spinner=False)
def get_cached_model_bundle():
    return load_model_bundle()


@st.cache_data(ttl=5, show_spinner=False)
def load_platform_data():
    connection = get_postgres_connection()
    try:
        bootstrap_platform_tables_if_enabled(connection)
        predictions = load_prediction_logs(connection)
        alerts_df = load_alerts(connection)
        governance_df = load_governance_decisions(connection)
        return predictions, alerts_df, governance_df
    finally:
        connection.close()


def clear_runtime_caches(clear_model_bundle: bool = False):
    load_platform_data.clear()
    load_manager_actions.clear()
    build_decision_support_frame.clear()
    if clear_model_bundle:
        get_cached_model_bundle.clear()


def format_currency_tnd(value: float | int, decimals: int = 0) -> str:
    numeric_value = _safe_float(value)
    return f"{numeric_value:,.{decimals}f} TND"


def get_github_training_dispatch_config() -> dict:
    return {
        'token': os.getenv('GITHUB_ACTIONS_TOKEN', '').strip(),
        'repository': os.getenv('GITHUB_REPOSITORY', '').strip(),
        'workflow': os.getenv('GITHUB_TRAINING_WORKFLOW', 'training.yml').strip(),
        'ref': os.getenv('GITHUB_WORKFLOW_REF', 'main').strip(),
    }


def get_github_monitoring_dispatch_config() -> dict:
    config = get_github_training_dispatch_config().copy()
    config['workflow'] = os.getenv('GITHUB_MONITORING_WORKFLOW', 'monitoring.yml').strip()
    return config


def get_automation_execution_mode() -> str:
    mode = os.getenv('AUTOMATION_EXECUTION_MODE', 'auto').strip().lower()
    return mode if mode in {'auto', 'github', 'local'} else 'auto'


def is_github_api_ready() -> bool:
    config = get_github_training_dispatch_config()
    return bool(config['token'] and config['repository'])


def is_github_training_dispatch_ready() -> bool:
    config = get_github_training_dispatch_config()
    return all([config['token'], config['repository'], config['workflow'], config['ref']])


def is_github_monitoring_dispatch_ready() -> bool:
    config = get_github_monitoring_dispatch_config()
    return all([config['token'], config['repository'], config['workflow'], config['ref']])


def build_github_api_headers() -> dict:
    config = get_github_training_dispatch_config()
    return {
        'Accept': 'application/vnd.github+json',
        'Authorization': f'Bearer {config["token"]}',
        'User-Agent': 'TeleLink-Streamlit-MLOps-App',
    }


def fetch_latest_github_workflow_run(workflow_file: str) -> dict | None:
    if not is_github_api_ready():
        return None

    config = get_github_training_dispatch_config()
    url = (
        f'https://api.github.com/repos/{config["repository"]}/actions/workflows/'
        f'{workflow_file}/runs?per_page=1'
    )
    request = urllib_request.Request(
        url,
        method='GET',
        headers=build_github_api_headers(),
    )

    try:
        with urllib_request.urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode('utf-8'))
        runs = payload.get('workflow_runs', [])
        if not runs:
            return {
                'workflow': workflow_file,
                'status': 'no_runs',
                'conclusion': '',
                'created_at': '',
                'html_url': '',
            }
        run = runs[0]
        return {
            'workflow': workflow_file,
            'status': run.get('status', 'unknown'),
            'conclusion': run.get('conclusion') or '',
            'created_at': run.get('created_at') or '',
            'html_url': run.get('html_url') or '',
            'event': run.get('event') or '',
            'run_number': run.get('run_number') or '',
        }
    except Exception as error:
        return {
            'workflow': workflow_file,
            'status': 'api_error',
            'conclusion': '',
            'created_at': '',
            'html_url': '',
            'error': str(error),
        }


def get_recent_github_workflow_runs() -> list[dict]:
    workflow_files = ['ci.yml', 'cd.yml', 'training.yml', 'monitoring.yml']
    runs = []
    for workflow in workflow_files:
        run = fetch_latest_github_workflow_run(workflow)
        if run is not None:
            runs.append(run)
    return runs


def trigger_github_workflow_dispatch(config: dict, inputs: dict, workflow_label: str) -> tuple[bool, str]:
    required_values = [config.get('token'), config.get('repository'), config.get('workflow'), config.get('ref')]
    if not all(required_values):
        return False, (
            f'GitHub Actions dispatch is not configured for {workflow_label}. '
            'Set GITHUB_ACTIONS_TOKEN, GITHUB_REPOSITORY, the workflow filename, and GITHUB_WORKFLOW_REF.'
        )

    owner_repo = config['repository']
    workflow = config['workflow']
    url = f'https://api.github.com/repos/{owner_repo}/actions/workflows/{workflow}/dispatches'
    payload = json.dumps(
        {
            'ref': config['ref'],
            'inputs': inputs,
        }
    ).encode('utf-8')
    request = urllib_request.Request(
        url,
        data=payload,
        method='POST',
        headers={
            'Accept': 'application/vnd.github+json',
            'Authorization': f'Bearer {config["token"]}',
            'Content-Type': 'application/json',
            'User-Agent': 'TeleLink-Streamlit-MLOps-App',
        },
    )

    try:
        with urllib_request.urlopen(request, timeout=20) as response:
            status_code = getattr(response, 'status', response.getcode())
        if status_code in (200, 201, 204):
            return True, (
                f'GitHub Actions {workflow_label} workflow "{workflow}" was dispatched successfully '
                f'for repository "{owner_repo}" on ref "{config["ref"]}".'
            )
        return False, f'GitHub API returned unexpected status code {status_code}.'
    except urllib_error.HTTPError as error:
        body = error.read().decode('utf-8', errors='ignore')
        if error.code == 422 and 'Unexpected inputs provided' in body:
            return False, (
                f'GitHub {workflow_label} workflow dispatch failed with HTTP 422: {body} '
                f'This usually means the workflow file "{workflow}" on ref "{config["ref"]}" '
                f'in repository "{owner_repo}" does not declare the inputs this app is sending yet. '
                'Push the updated workflow file to that branch/ref, confirm the workflow filename matches, '
                'or change GITHUB_WORKFLOW_REF to the branch that already contains the new workflow_dispatch inputs.'
            )
        return False, f'GitHub {workflow_label} workflow dispatch failed with HTTP {error.code}: {body or error.reason}'
    except urllib_error.URLError as error:
        return False, f'Unable to reach GitHub Actions API: {error.reason}'
    except Exception as error:
        return False, f'Unexpected GitHub {workflow_label} workflow dispatch error: {error}'


def trigger_github_training_workflow(reason: str, profile: str = 'quick') -> tuple[bool, str]:
    return trigger_github_workflow_dispatch(
        config=get_github_training_dispatch_config(),
        inputs={
            'reason': reason,
            'profile': profile,
        },
        workflow_label='training',
    )


def trigger_github_monitoring_workflow(
    days: int = 14,
    threshold: int = 2,
    auto_retrain: bool = True,
    profile: str = 'quick',
) -> tuple[bool, str]:
    return trigger_github_workflow_dispatch(
        config=get_github_monitoring_dispatch_config(),
        inputs={
            'days': str(days),
            'threshold': str(threshold),
            'auto_retrain': 'true' if auto_retrain else 'false',
            'profile': profile,
        },
        workflow_label='monitoring',
    )


def launch_local_training_pipeline(reason: str, profile: str = 'quick') -> tuple[bool, str]:
    command = [
        sys.executable,
        str(ROOT_DIR / 'Scripts' / 'run_training.py'),
        '--reason',
        reason,
        '--profile',
        profile,
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    combined_output = '\n'.join(part for part in [completed.stdout, completed.stderr] if part).strip()
    return completed.returncode == 0, combined_output


def launch_training_pipeline(reason: str, profile: str = 'quick', execution_mode: str = 'auto') -> tuple[bool, str]:
    execution_mode = execution_mode or get_automation_execution_mode()
    if execution_mode == 'github':
        return trigger_github_training_workflow(reason=reason, profile=profile)
    if execution_mode == 'local':
        return launch_local_training_pipeline(reason=reason, profile=profile)
    if is_github_training_dispatch_ready():
        return trigger_github_training_workflow(reason=reason, profile=profile)
    return launch_local_training_pipeline(reason=reason, profile=profile)


def launch_monitoring_pipeline(
    days: int = 14,
    threshold: int = 2,
    auto_retrain: bool = True,
    profile: str = 'quick',
    execution_mode: str = 'auto',
) -> tuple[bool, str]:
    execution_mode = execution_mode or get_automation_execution_mode()
    if execution_mode == 'github':
        return trigger_github_monitoring_workflow(
            days=days,
            threshold=threshold,
            auto_retrain=auto_retrain,
            profile=profile,
        )
    if execution_mode == 'auto' and is_github_monitoring_dispatch_ready():
        return trigger_github_monitoring_workflow(
            days=days,
            threshold=threshold,
            auto_retrain=auto_retrain,
            profile=profile,
        )
    return False, 'Remote monitoring dispatch is disabled because the selected execution mode is not GitHub.'


def count_high_severity_alerts(alerts: list[dict]) -> int:
    return sum(1 for alert in alerts if alert.get('severity') == 'high')


def prepare_monitoring_dataframe(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    df = predictions.copy()
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    return df.sort_values('created_at', ascending=False)


# --- UI Helper Functions ---
def get_prediction_source_label(source: str | None) -> str:
    source_map = {
        'demo_seed': 'Demo Seeded',
        FORM_SOURCE: 'Manual Entry',
    }
    return source_map.get((source or '').strip(), 'Other Source')


def add_prediction_source_columns(df: pd.DataFrame) -> pd.DataFrame:
    enriched_df = df.copy()
    source_series = enriched_df.get('source', pd.Series(index=enriched_df.index, dtype='object')).fillna('')
    enriched_df['Prediction Source'] = source_series.map(get_prediction_source_label)
    enriched_df['Is Demo Record'] = enriched_df['Prediction Source'].eq('Demo Seeded')
    return enriched_df


def render_decision_badges(action_guidance: dict):
    value_class = {
        'High value': 'value-high',
        'Medium value': 'value-medium',
        'Standard value': 'value-standard',
    }.get(action_guidance['value_tier'], 'value-standard')
    severity_class = {
        'High severity': 'severity-high',
        'Medium severity': 'severity-medium',
        'Lower severity': 'severity-lower',
    }.get(action_guidance['severity_tier'], 'severity-lower')
    owner = action_guidance.get('execution_owner', 'Retention Team')
    if owner == 'Account Manager':
        owner_class = 'action-escalate'
    elif 'Support' in owner:
        owner_class = 'action-offer'
    else:
        owner_class = 'action-call'

    st.markdown(
        (
            '<div class="decision-badge-row">'
            f'<span class="decision-badge {value_class}">Value: {action_guidance["value_tier"]}</span>'
            f'<span class="decision-badge {severity_class}">Severity: {action_guidance["severity_tier"]}</span>'
            f'<span class="decision-badge {owner_class}">Owner: {owner}</span>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )


def get_repo_health_snapshot() -> dict:
    workflow_dir = ROOT_DIR / '.github' / 'workflows'
    workflow_files = {
        'CI': workflow_dir / 'ci.yml',
        'CD': workflow_dir / 'cd.yml',
        'Monitoring': workflow_dir / 'monitoring.yml',
        'Training': workflow_dir / 'training.yml',
    }
    bundle_paths = [
        ROOT_DIR / 'models' / 'churn_production_bundle.pkl',
        ROOT_DIR / 'churn_production.pkl',
    ]
    available_bundles = [path for path in bundle_paths if path.exists()]
    latest_bundle = max(available_bundles, key=lambda path: path.stat().st_mtime) if available_bundles else None

    return {
        'git_repo_present': (ROOT_DIR / '.git').exists(),
        'workflows': {name: path.exists() for name, path in workflow_files.items()},
        'has_alembic': (ROOT_DIR / 'alembic.ini').exists() and (ROOT_DIR / 'alembic').exists(),
        'bundle_count': len(available_bundles),
        'latest_bundle_name': latest_bundle.name if latest_bundle else None,
    }


def get_runtime_configuration_snapshot() -> pd.DataFrame:
    github_dispatch_ready = is_github_training_dispatch_ready()
    github_monitoring_ready = is_github_monitoring_dispatch_ready()
    github_config = get_github_training_dispatch_config()
    monitoring_config = get_github_monitoring_dispatch_config()
    rows = [
        {'Setting': 'MLflow Tracking URI', 'Value': get_mlflow_tracking_uri()},
        {'Setting': 'MLflow Experiment', 'Value': get_mlflow_experiment_name()},
        {'Setting': 'Postgres Host', 'Value': os.getenv('POSTGRES_HOST', 'db')},
        {'Setting': 'Postgres Port', 'Value': os.getenv('POSTGRES_PORT', '5432')},
        {'Setting': 'Automation Execution Mode', 'Value': get_automation_execution_mode()},
        {'Setting': 'GitHub Training Dispatch', 'Value': 'Ready' if github_dispatch_ready else 'Not Configured'},
        {'Setting': 'GitHub Monitoring Dispatch', 'Value': 'Ready' if github_monitoring_ready else 'Not Configured'},
        {'Setting': 'GitHub Repository', 'Value': github_config['repository'] or 'Not set'},
        {'Setting': 'GitHub Training Workflow', 'Value': github_config['workflow'] or 'Not set'},
        {'Setting': 'GitHub Monitoring Workflow', 'Value': monitoring_config['workflow'] or 'Not set'},
        {'Setting': 'GitHub Workflow Ref', 'Value': github_config['ref'] or 'Not set'},
        {'Setting': 'Env Example Present', 'Value': 'Yes' if (ROOT_DIR / '.env.example').exists() else 'No'},
    ]
    return pd.DataFrame(rows)


def render_mlops_sidebar_status():
    snapshot = get_repo_health_snapshot()
    workflow_count = sum(snapshot['workflows'].values())
    st.markdown('### Technical Status')
    metric_cols = st.columns(2)
    metric_cols[0].metric('Workflows', f'{workflow_count}/4')
    metric_cols[1].metric('Bundles', snapshot['bundle_count'])
    st.caption(
        f"Alembic: {'Ready' if snapshot['has_alembic'] else 'Missing'} | "
        f"Git repo: {'Detected' if snapshot['git_repo_present'] else 'Missing'}"
    )
    with st.expander('Workflow File Check'):
        for name, is_present in snapshot['workflows'].items():
            st.write(f"- `{name}`: {'Present' if is_present else 'Missing'}")


def render_mlops_story_panel():
    st.markdown('### End-To-End MLOps Flow')
    steps = pd.DataFrame(
        [
            {'Step': '1. Predict', 'What Happens': 'The production model scores a customer and explains the main churn drivers.'},
            {'Step': '2. Decide', 'What Happens': 'The app recommends a next best action and the manager records the chosen response.'},
            {'Step': '3. Log', 'What Happens': 'Predictions, actions, and outcomes are stored in PostgreSQL for traceability.'},
            {'Step': '4. Monitor', 'What Happens': 'Live predictions are reviewed for alerts, drift, feedback coverage, and performance change.'},
            {'Step': '5. Retrain', 'What Happens': 'If alerts cross the threshold, the training pipeline creates a new candidate model locally.'},
            {'Step': '6. Govern', 'What Happens': 'The candidate is compared with the production baseline and the governance decision is recorded.'},
            {'Step': '7. Refresh Production', 'What Happens': 'The approved bundle becomes the active production model for the next monitoring cycle.'},
        ]
    )
    st.dataframe(steps, width='stretch', hide_index=True)


def render_delivery_readiness_panel():
    snapshot = get_repo_health_snapshot()
    runtime_df = get_runtime_configuration_snapshot()

    st.markdown('### CI/CD And Registry Readiness')
    readiness_cols = st.columns(5)
    readiness_cols[0].metric('CI', 'Ready' if snapshot['workflows'].get('CI') else 'Missing')
    readiness_cols[1].metric('CD', 'Ready' if snapshot['workflows'].get('CD') else 'Missing')
    readiness_cols[2].metric('Migrations', 'Ready' if snapshot['has_alembic'] else 'Missing')
    readiness_cols[3].metric('Bundles', snapshot['bundle_count'])
    readiness_cols[4].metric('Workflow Evidence', 'Ready' if snapshot['git_repo_present'] else 'Local Only')

    if snapshot['latest_bundle_name']:
        st.caption(f"Latest local production bundle: `{snapshot['latest_bundle_name']}`")
    else:
        st.caption('No production bundle found yet. Run the training pipeline to generate one.')

    info_col, table_col = st.columns([0.95, 1.05])
    with info_col:
        st.info(
            'This panel helps you verify whether the repo contains the pieces expected for GitHub CI/CD, '
            'remote automation evidence, local demo execution, and release readiness.'
        )
        if not snapshot['git_repo_present']:
            st.warning('No local `.git` directory was detected in this workspace. Push the project from a real Git repository to activate GitHub Actions.')
    with table_col:
        st.dataframe(runtime_df, width='stretch', hide_index=True)


def render_cicd_story_panel():
    snapshot = get_repo_health_snapshot()
    st.markdown('### CI/CD Overview')
    st.caption(
        'This section summarizes how validation, deployment, monitoring, and retraining are connected in the project workflow.'
    )

    summary_cols = st.columns(4)
    summary_cols[0].metric('CI', 'Active' if snapshot['workflows'].get('CI') else 'Missing')
    summary_cols[1].metric('CD', 'Active' if snapshot['workflows'].get('CD') else 'Missing')
    summary_cols[2].metric('Monitoring Job', 'Active' if snapshot['workflows'].get('Monitoring') else 'Missing')
    retraining_path = 'Local Demo Path'
    summary_cols[3].metric('Retraining Path', retraining_path)

    flow_cols = st.columns(4)
    flow_cols[0].info(
        '1. Build Quality\n\n'
        'Every push or pull request goes through CI checks: linting, test execution, coverage, and Docker build validation.'
    )
    flow_cols[1].info(
        '2. Delivery\n\n'
        'After validation, CD packages the application into a deployable container so the delivered version matches a reviewed Git commit.'
    )
    flow_cols[2].info(
        '3. Operations\n\n'
        'This Streamlit app logs production evidence to PostgreSQL, and the monitoring workflow turns that evidence into health alerts.'
    )
    flow_cols[3].info(
        '4. Continuous Improvement\n\n'
        'The project includes GitHub workflow automation, while the live app demo runs retraining locally for reliability and still tracks outcomes in MLflow.'
    )

    benefit_cols = st.columns(3)
    benefit_cols[0].success(
        'Reproducibility\n\n'
        'Training, testing, and packaging run the same way each time instead of depending on one machine.'
    )
    benefit_cols[1].success(
        'Auditability\n\n'
        'Model changes, monitoring runs, and release steps leave a visible execution trail for reviewers and teams.'
    )
    benefit_cols[2].success(
        'Production Credibility\n\n'
        'The project demonstrates an operational ML system, not only a prediction interface or notebook result.'
    )

    render_delivery_readiness_panel()


def render_automation_quick_guide():
    st.markdown('### Automation Flow')
    guide_cols = st.columns(3)
    guide_cols[0].info(
        '1. CI\n\nPush the repo to GitHub. Every push or pull request will run lint, tests, coverage, and Docker build validation.'
    )
    guide_cols[1].info(
        '2. Training\n\nUse the GitHub Training workflow as automation evidence, and use the in-app local training path during the jury demo to create updated MLflow runs and bundles safely.'
    )
    guide_cols[2].info(
        '3. Monitoring\n\nUse the GitHub Monitoring workflow as architecture evidence, and use the in-app local monitoring cycle during the jury demo to evaluate production evidence and trigger local retraining when needed.'
    )


def render_header():
    hero_image_path = ROOT_DIR / 'assets' / 'telco_churn_prediction.png'
    if hero_image_path.exists():
        hero_image_base64 = base64.b64encode(hero_image_path.read_bytes()).decode('utf-8')
        st.markdown(
            f"""
            <div style="width:100%;margin-bottom:1rem;">
                <img
                    src="data:image/png;base64,{hero_image_base64}"
                    alt="TelCo Churn dashboard hero"
                    style="width:100%;max-height:500px;object-fit:cover;object-position:center 46%;display:block;border-radius:18px;"
                />
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown(
        """
        <div class="hero-title-wrap">
            <div class="hero-title-icon">&#128202;</div>
            <div class="hero-title-copy">
                <div class="main-header">TelCo Churn Prediction</div>
                <div class="hero-title-subtitle">Managerial Decision Support System</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption('Use the dashboard to identify churn risk, prioritize retention actions, monitor model health, and decide when retraining is needed.')
    st.divider()


def format_feature_name(feature: str) -> str:
    replacements = {
        'tenure': 'Tenure',
        'MonthlyCharges': 'Monthly Charges',
        'Data_Usage_GB': 'Data Usage',
        'TotalServices': 'Total Services',
        'Total_Revenue': 'Total Revenue',
        'Usage_per_Month': 'Usage/Month',
        'gender': 'Gender',
        'SeniorCitizen': 'Senior Citizen',
        'Contract': 'Contract Type',
        'InternetService': 'Internet Service',
        'PaymentMethod': 'Payment Method',
        'PaperlessBilling': 'Paperless Billing',
        'Gouvernorat': 'Governorate',
        'Dependents': 'Dependents',
        'OnlineSecurity': 'Online Security',
        'OnlineBackup': 'Online Backup',
        'DeviceProtection': 'Device Protection',
        'StreamingTV': 'Streaming TV',
        'StreamingMovies': 'Streaming Movies',
        'Support_Tickets': 'Support Tickets',
        'App_Logins': 'App Logins',
    }
    return replacements.get(feature, feature.replace('_', ' ').title())


def format_monitoring_alert(alert: dict) -> str:
    alert_type = alert.get('alert_type', 'unknown')
    if alert_type == 'accuracy_drop':
        return f"Accuracy dropped by {alert.get('value', 0):.1%} (threshold: {alert.get('threshold', 0):.1%})"
    if alert_type == 'f1_drop':
        return f"F1-score dropped by {alert.get('value', 0):.1%} (threshold: {alert.get('threshold', 0):.1%})"
    if alert_type == 'probability_drift':
        return f"Probability drift detected: {alert.get('value', 0):.3f} shift"
    if alert_type == 'high_risk_increase':
        return f"High-risk customers increased by {alert.get('value', 0):.1%}"
    if alert_type == 'low_ground_truth_coverage':
        return f"Ground-truth coverage too low: {alert.get('value', 0):.1%} (minimum: {alert.get('threshold', 0):.1%})"
    return f"{alert_type}: {alert.get('value', 'N/A')}"


def evaluate_governance_candidate(candidate_metrics: dict, baseline_metrics: dict) -> dict:
    accuracy_delta = candidate_metrics['accuracy'] - baseline_metrics['accuracy']
    f1_delta = candidate_metrics['f1'] - baseline_metrics['f1']
    roc_auc_delta = candidate_metrics['roc_auc'] - baseline_metrics['roc_auc']

    if accuracy_delta >= 0 and f1_delta >= 0 and roc_auc_delta >= 0:
        decision = 'APPROVED'
        rationale = 'Candidate outperforms or matches production baseline across all metrics.'
    elif accuracy_delta >= -0.02 and f1_delta >= -0.02 and roc_auc_delta >= -0.02:
        decision = 'CONDITIONAL APPROVAL'
        rationale = 'Candidate within 2% of baseline. Consider deployment if F1 improves.'
    else:
        decision = 'REJECTED'
        rationale = 'Candidate underperforms production baseline. Retrain with different hyperparameters.'

    return {'decision': decision, 'rationale': rationale}


# --- Customer Input Functions ---
def build_customer_inputs(prefix: str = 'customer') -> dict:
    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox('Gender', ['Female', 'Male'], key=f'{prefix}_gender')
        senior_citizen = st.selectbox('Senior Citizen', ['No', 'Yes'], key=f'{prefix}_senior')
        tenure = st.number_input('Tenure (months)', min_value=0, max_value=72, value=12, key=f'{prefix}_tenure')
        monthly_charges = st.number_input('Monthly Charges (TND)', min_value=0.0, max_value=200.0, value=70.0, key=f'{prefix}_charges')
    with col2:
        contract = st.selectbox('Contract', ['month-to-month', 'one year', 'two year'], key=f'{prefix}_contract')
        data_usage = st.number_input('Data Usage (GB)', min_value=0.0, max_value=100.0, value=10.0, key=f'{prefix}_data')
        support_tickets = st.number_input('Support Tickets', min_value=0, max_value=20, value=1, key=f'{prefix}_tickets')
        app_logins = st.number_input('App Logins', min_value=0, max_value=50, value=5, key=f'{prefix}_logins')
    with col3:
        internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'None'], key=f'{prefix}_internet')
        payment_method = st.selectbox(
            'Payment Method',
            ['Electronic Check', 'Mailed Check', 'Bank Transfer (automatic)', 'Credit Card (automatic)'],
            key=f'{prefix}_payment',
        )
        paperless_billing = st.selectbox('Paperless Billing', ['Yes', 'No'], key=f'{prefix}_paperless')
        gouvernorat = st.selectbox(
            'Governorate',
            TUNISIA_GOVERNORATES,
            key=f'{prefix}_gov',
        )

    col4, col5, col6 = st.columns(3)
    with col4:
        dependents = st.selectbox('Dependents', ['No', 'Yes'], key=f'{prefix}_dependents')
        online_security = st.selectbox('Online Security', ['No', 'Yes'], key=f'{prefix}_security')
    with col5:
        online_backup = st.selectbox('Online Backup', ['No', 'Yes'], key=f'{prefix}_backup')
        device_protection = st.selectbox('Device Protection', ['No', 'Yes'], key=f'{prefix}_device')
    with col6:
        streaming_tv = st.selectbox('Streaming TV', ['No', 'Yes'], key=f'{prefix}_tv')
        streaming_movies = st.selectbox('Streaming Movies', ['No', 'Yes'], key=f'{prefix}_movies')

    return {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'Contract': contract,
        'Data_Usage_GB': data_usage,
        'Support_Tickets': support_tickets,
        'App_Logins': app_logins,
        'PaymentMethod': payment_method,
        'InternetService': internet_service,
        'PaperlessBilling': paperless_billing,
        'Gouvernorat': gouvernorat,
        'Dependents': dependents,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
    }


# --- Next Best Action Logic ---
def get_next_best_action(risk_label: str) -> str:
    config = load_action_logic_config()
    action_name = config['risk_actions'].get(risk_label, DEFAULT_RISK_ACTION_CONFIG.get(risk_label, 'No Action Needed'))
    return normalize_action_label(action_name)


def get_manager_action_options(risk_label: str) -> list[str]:
    options_map = {
        'HIGH RISK': [
            'Phone Call - Priority 1',
            'Immediate Retention Call + 15% Discount Offer',
            'Retention Call + 10% Discount Offer',
            'Retention Call + Gold Data Pack Offer',
            'Account Manager Call + Gold Data Pack Offer',
            'Escalate to Service Recovery Team',
            'Account Manager + Service Recovery Escalation',
            'Retention Call + Contract Upgrade Offer',
            'Retention Call + Premium Contract Upgrade Offer',
            'Account Manager Call + Premium Contract Upgrade Offer',
            'Retention Call + Security/Backup Bundle',
            'Retention Call + Premium Bundle Offer',
            'Account Manager Call + Premium Bundle Offer',
            'Retention Call + Usage Coaching',
            'Retention Call + Reactivation Offer',
            'Account Manager Call + Reactivation Offer',
            'Account Manager Call + 15% Discount Offer',
        ],
        'MEDIUM RISK': [
            'SMS Discount Offer',
            'Nurture Campaign + Check-in Call',
            'Personalized Email Offer',
            'Schedule Follow-up Review',
            'Personalized Check-In Email',
            'SMS Offer - Silver Data Pack',
            'SMS Offer - Bronze Data Pack',
            'Retention Call + Service Recovery',
            'Personalized Contract Upgrade Offer',
            'Renewal Reminder + Value Email',
            'Personalized Bill Relief Offer',
            'Personalized Savings Email',
            'Personalized Add-on Bundle Offer',
            'Add-on Discovery Email',
            'Onboarding Check-in Journey',
            'Welcome Check-In Email',
            'Usage Tips Email',
        ],
        'LOW RISK': [
            'No Action Needed',
            'Monitor Only',
            'Personalized Check-In Email',
            'Monitor Data-Pack Eligibility',
            'Monitor Renewal Window',
            'Monitor Bundle Adoption',
        ],
    }
    normalized_options = [normalize_action_label(option) for option in options_map.get(risk_label, ['No Action Needed'])]
    return list(dict.fromkeys(normalized_options))


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _get_driver_impact_map(top_drivers) -> dict[str, float]:
    if not top_drivers:
        return {}
    impact_map = {}
    for driver in top_drivers:
        if not isinstance(driver, dict):
            continue
        feature = str(driver.get('feature') or '').strip()
        if not feature:
            continue
        impact_map[feature] = abs(_safe_float(driver.get('impact')))
    return impact_map


def _format_model_driver_label(feature_name: str) -> str:
    feature_name = str(feature_name or '').strip()
    if not feature_name:
        return 'Unknown factor'

    mapped_labels = {
        'MonthlyCharges': 'Monthly charges are high',
        'Data_Usage_GB': 'Data usage is high',
        'Support_Tickets': 'Support tickets are elevated',
        'App_Logins': 'App engagement is low',
        'Contract': 'Contract setup lowers commitment',
        'tenure': 'Tenure is still short',
        'PaymentMethod': 'Payment method may add friction',
        'OnlineSecurity': 'Online security is missing',
        'OnlineBackup': 'Online backup is missing',
        'InternetService': 'Internet service profile may be vulnerable',
    }
    if feature_name in mapped_labels:
        return mapped_labels[feature_name]

    if '=' in feature_name:
        base, value = feature_name.split('=', 1)
        base = base.strip()
        value = value.strip()
        mapped_base = {
            'Contract': 'Contract type',
            'PaymentMethod': 'Payment method',
            'InternetService': 'Internet service',
            'OnlineSecurity': 'Online security',
            'OnlineBackup': 'Online backup',
            'PaperlessBilling': 'Paperless billing',
        }.get(base, base.replace('_', ' ').title())
        return f'{mapped_base}: {value}'

    return feature_name.replace('_', ' ').title()


def build_customer_driver_story(row: pd.Series, top_drivers=None) -> dict:
    risk_label = str(row.get('predicted_risk') or 'LOW RISK').strip().upper() or 'LOW RISK'
    reasons = get_business_churn_reasons(row, top_drivers=top_drivers)
    active_top_drivers = top_drivers or row.get('top_drivers') or []

    top_driver_labels = []
    for driver in active_top_drivers:
        if not isinstance(driver, dict):
            continue
        label = _format_model_driver_label(driver.get('feature'))
        if label and label not in top_driver_labels:
            top_driver_labels.append(label)

    if not top_driver_labels:
        top_driver_labels = [reason['label'] for reason in reasons[:3]]

    if risk_label == 'HIGH RISK':
        watchout_title = 'Why this customer may leave'
        watchout_text = 'The risk is immediate, so the action should remove the main friction now.'
    elif risk_label == 'MEDIUM RISK':
        watchout_title = 'Main churn warning signals'
        watchout_text = 'The signals are building, so the action should prevent the case from escalating.'
    else:
        watchout_title = 'Main watch-out factors'
        watchout_text = 'The customer is still relatively stable, so the action should stay light and preventive.'

    return {
        'reasons': reasons,
        'top_driver_labels': top_driver_labels[:3],
        'headline_reason': reasons[0]['label'] if reasons else 'No clear reason',
        'watchout_title': watchout_title,
        'watchout_text': watchout_text,
    }


def get_playbook_template(category: str) -> dict:
    config = load_action_logic_config()
    playbooks = config.get('playbooks', {})
    return playbooks.get(category, playbooks.get('general', DEFAULT_ACTION_PLAYBOOKS['general']))


def get_probability_decile_label(probability: float) -> str:
    probability = max(0.0, min(1.0, probability))
    upper_bound = int(np.ceil(probability * 10) * 10)
    if upper_bound == 0:
        upper_bound = 10
    lower_bound = upper_bound - 9
    return f'{lower_bound}-{upper_bound}%'


def get_probability_decile_action(probability: float, category: str, value_tier: str = 'Standard value') -> dict:
    probability = max(0.0, min(1.0, probability))
    category = category if category in DEFAULT_ACTION_PLAYBOOKS else 'general'
    is_high_value = value_tier == 'High value'
    category_actions = {
        'low_risk': {
            'data_usage': [
                (0.10, 'Customer Loyalty', 'Loyalty Thank-You Message', 'Very low churn probability calls for appreciation, not intervention.', 'Retention Team'),
                (0.20, 'Usage Watch', 'Monitor Data-Pack Eligibility', 'Risk is still low, so watch for a better-fit bundle before making a stronger offer.', 'Retention Team'),
                (0.30, 'Usage Watch', 'Usage Tips Email', 'Usage is rising, so a light guidance touch can improve value perception early.', 'Retention Team'),
                (0.40, 'Data Usage Retention', 'SMS Offer - Bronze Data Pack', 'Some early warning signals suggest a small bundle correction could help.', 'Retention Team'),
                (0.50, 'Data Usage Retention', 'SMS Offer - Bronze Data Pack', 'A light data-pack offer is appropriate before stronger intervention.', 'Retention Team'),
            ],
            'billing': [
                (0.10, 'Customer Loyalty', 'Loyalty Thank-You Message', 'Very low churn probability calls for appreciation, not intervention.', 'Retention Team'),
                (0.20, 'Price Watch', 'Personalized Savings Email', 'Risk is still low, but a light value reminder can ease early price sensitivity.', 'Retention Team'),
                (0.30, 'Price Watch', 'Personalized Savings Email', 'Pricing pressure is visible, so a soft savings message is enough for now.', 'Retention Team'),
                (0.40, 'Price Protection', 'Personalized Bill Relief Offer', 'Some early warning signals suggest a more tailored value message is justified.', 'Retention Team'),
                (0.50, 'Price Protection', 'Personalized Bill Relief Offer', 'A light retention value offer is appropriate before stronger intervention.', 'Retention Team'),
            ],
            'support': [
                (0.10, 'Customer Loyalty', 'Loyalty Thank-You Message', 'Very low churn probability calls for appreciation, not intervention.', 'Retention Team'),
                (0.20, 'Service Watch', 'Service Check-In SMS', 'Risk is still low, so a quick reassurance touch is enough.', 'Retention Team'),
                (0.30, 'Service Watch', 'Priority Support Follow-up', 'Service signals are visible and should be handled early before they grow.', 'Service Recovery Team'),
                (0.40, 'Service Recovery', 'Priority Support Follow-up', 'Some early warning signals suggest active service follow-up is warranted.', 'Service Recovery Team'),
                (0.50, 'Service Recovery', 'Retention Call + Service Recovery', 'A direct recovery touch is appropriate before the case escalates further.', 'Service Recovery Team'),
            ],
            'engagement': [
                (0.10, 'Customer Loyalty', 'Loyalty Thank-You Message', 'Very low churn probability calls for appreciation, not intervention.', 'Retention Team'),
                (0.20, 'Light Engagement', 'Usage Tips Email', 'Risk is still low, so a soft reactivation nudge is enough.', 'Retention Team'),
                (0.30, 'Light Engagement', 'Personalized Check-In Email', 'Engagement is softening, so a personalized relationship touch is appropriate.', 'Retention Team'),
                (0.40, 'Customer Reactivation', 'Nurture Campaign + Check-in Call', 'Some early warning signals suggest a guided reactivation touch is justified.', 'Retention Team'),
                (0.50, 'Customer Reactivation', 'Nurture Campaign + Check-in Call', 'A more active reactivation touch is appropriate before stronger intervention.', 'Retention Team'),
            ],
            'contract': [
                (0.10, 'Customer Loyalty', 'Loyalty Thank-You Message', 'Very low churn probability calls for appreciation, not intervention.', 'Retention Team'),
                (0.20, 'Contract Watch', 'Monitor Renewal Window', 'Risk is still low, so watch the renewal timing before making a stronger save offer.', 'Retention Team'),
                (0.30, 'Contract Watch', 'Renewal Reminder + Value Email', 'The contract setup is a watch-out factor, so a light value reminder is enough for now.', 'Retention Team'),
                (0.40, 'Contract Conversion', 'Personalized Contract Upgrade Offer', 'Some early warning signals suggest a stickier-plan offer is justified.', 'Retention Team'),
                (0.50, 'Contract Conversion', 'Personalized Contract Upgrade Offer', 'A light upgrade offer is appropriate before stronger intervention.', 'Retention Team'),
            ],
            'tenure': [
                (0.10, 'Customer Loyalty', 'Loyalty Thank-You Message', 'Very low churn probability calls for appreciation, not intervention.', 'Retention Team'),
                (0.20, 'Early Lifecycle Retention', 'Welcome Check-In Email', 'Risk is still low, but a welcome reassurance touch can reinforce early habits.', 'Retention Team'),
                (0.30, 'Early Lifecycle Retention', 'Onboarding Check-in Journey', 'This newer relationship needs a little more structure to build attachment.', 'Retention Team'),
                (0.40, 'Early Lifecycle Retention', 'Onboarding Check-in Journey', 'Some early warning signals suggest a more guided onboarding touch is justified.', 'Retention Team'),
                (0.50, 'Early Lifecycle Retention', 'Retention Call + Welcome Benefit', 'A direct save touch is appropriate before the case escalates further.', 'Retention Team'),
            ],
            'bundle': [
                (0.10, 'Customer Loyalty', 'Loyalty Thank-You Message', 'Very low churn probability calls for appreciation, not intervention.', 'Retention Team'),
                (0.20, 'Bundle Watch', 'Monitor Bundle Adoption', 'Risk is still low, so monitor whether the customer adopts sticky add-ons.', 'Retention Team'),
                (0.30, 'Bundle Watch', 'Add-on Discovery Email', 'Bundle attachment looks light, so a soft discovery message is enough for now.', 'Retention Team'),
                (0.40, 'Bundle Attachment', 'Personalized Add-on Bundle Offer', 'Some early warning signals suggest a more tailored add-on offer is justified.', 'Retention Team'),
                (0.50, 'Bundle Attachment', 'Personalized Add-on Bundle Offer', 'A light bundle offer is appropriate before stronger intervention.', 'Retention Team'),
            ],
            'general': [
                (0.10, 'Customer Loyalty', 'Loyalty Thank-You Message', 'Very low churn probability calls for appreciation, not intervention.', 'Retention Team'),
                (0.20, 'Light Engagement', 'Personalized Check-In Email', 'Risk is still low, so a soft relationship touch is enough.', 'Retention Team'),
                (0.30, 'Light Monitoring', 'Monitor Only', 'This customer is still low risk and only needs light observation.', 'Retention Team'),
                (0.40, 'Retention Watchlist', 'Schedule Follow-up Review', 'Some early warning signals are visible, but not enough for an offer yet.', 'Retention Team'),
                (0.50, 'Soft Retention', 'Personalized Email Offer', 'A light personalized offer is appropriate before stronger intervention.', 'Retention Team'),
            ],
        },
        'data_usage': [
            (0.60, 'Data Usage Retention', 'SMS Offer - Bronze Data Pack', 'Usage pressure is emerging, so start with a light data-pack promotion.', 'Retention Team'),
            (0.70, 'Data Usage Retention', 'SMS Offer - Silver Data Pack', 'The risk is rising, so a stronger data-pack value message is justified.', 'Retention Team'),
            (0.80, 'Data Usage Retention', 'Retention Call + Silver Data Pack Offer', 'A direct conversation plus a better-fit pack is appropriate now.', 'Retention Team'),
            (0.90, 'Data Usage Retention', 'Retention Call + Gold Data Pack Offer', 'Heavy usage is now a strong churn driver, so the top-value pack should be offered.', 'Retention Team'),
            (1.00, 'Data Usage Retention', 'Account Manager Call + Gold Data Pack Offer', 'Very high risk and high usage justify the strongest save path.', 'Account Manager' if is_high_value else 'Retention Team'),
        ],
        'billing': [
            (0.60, 'Price Protection', 'Personalized Savings Email', 'Price pressure is building, but an email save offer is still enough.', 'Retention Team'),
            (0.70, 'Price Protection', 'SMS Discount Offer', 'A clearer price-relief signal is needed at this stage.', 'Retention Team'),
            (0.80, 'Price Protection', 'Retention Call + 10% Discount Offer', 'A moderate retention discount is now justified.', 'Retention Team'),
            (0.90, 'Price Protection', 'Immediate Retention Call + 15% Discount Offer', 'Price sensitivity is now strong enough for a deeper save attempt.', 'Retention Team'),
            (1.00, 'Price Protection', 'Account Manager Call + 15% Discount Offer', 'Very high risk warrants a senior-led price save attempt.', 'Account Manager' if is_high_value else 'Retention Team'),
        ],
        'support': [
            (0.60, 'Service Recovery', 'Service Check-In SMS', 'Service friction is mild, so start with a quick reassurance touch.', 'Retention Team'),
            (0.70, 'Service Recovery', 'Priority Support Follow-up', 'Problems need active follow-up before they grow into churn.', 'Service Recovery Team'),
            (0.80, 'Service Recovery', 'Retention Call + Service Recovery', 'A direct recovery call is now the safest next step.', 'Service Recovery Team'),
            (0.90, 'Service Recovery', 'Escalate to Service Recovery Team', 'Service issues are now serious enough for formal escalation.', 'Service Recovery Team'),
            (1.00, 'Service Recovery', 'Account Manager + Service Recovery Escalation', 'High-value severe friction needs coordinated senior recovery.', 'Account Manager' if is_high_value else 'Service Recovery Team'),
        ],
        'engagement': [
            (0.60, 'Customer Reactivation', 'Usage Tips Email', 'A light reactivation nudge is enough at this stage.', 'Retention Team'),
            (0.70, 'Customer Reactivation', 'Nurture Campaign + Check-in Call', 'Engagement is slipping enough to justify a guided reactivation touch.', 'Retention Team'),
            (0.80, 'Customer Reactivation', 'Retention Call + Usage Coaching', 'A direct coaching conversation can rebuild product attachment.', 'Retention Team'),
            (0.90, 'Customer Reactivation', 'Retention Call + Reactivation Offer', 'Low engagement is now a strong churn warning and needs a stronger save offer.', 'Retention Team'),
            (1.00, 'Customer Reactivation', 'Account Manager Call + Reactivation Offer', 'Very high risk requires a senior-led reactivation attempt.', 'Account Manager' if is_high_value else 'Retention Team'),
        ],
        'contract': [
            (0.60, 'Contract Conversion', 'Renewal Reminder + Value Email', 'A soft reminder is enough before using a stronger contract offer.', 'Retention Team'),
            (0.70, 'Contract Conversion', 'Personalized Contract Upgrade Offer', 'The customer now needs a clearer value case for staying longer.', 'Retention Team'),
            (0.80, 'Contract Conversion', 'Retention Call + Contract Upgrade Offer', 'A direct call is appropriate to move the customer into a stickier plan.', 'Retention Team'),
            (0.90, 'Contract Conversion', 'Retention Call + Premium Contract Upgrade Offer', 'Higher urgency justifies a richer contract conversion offer.', 'Retention Team'),
            (1.00, 'Contract Conversion', 'Account Manager Call + Premium Contract Upgrade Offer', 'Very high-risk contract cases need a senior-led conversion push.', 'Account Manager' if is_high_value else 'Retention Team'),
        ],
        'tenure': [
            (0.60, 'Early Lifecycle Retention', 'Welcome Check-In Email', 'This is still an early but light save moment.', 'Retention Team'),
            (0.70, 'Early Lifecycle Retention', 'Onboarding Check-in Journey', 'The customer needs a structured early-life reassurance flow.', 'Retention Team'),
            (0.80, 'Early Lifecycle Retention', 'Retention Call + Welcome Benefit', 'A direct save touch is now safer for a newer customer.', 'Retention Team'),
            (0.90, 'Early Lifecycle Retention', 'Retention Call + Premium Welcome Benefit', 'Stronger urgency justifies a richer onboarding save offer.', 'Retention Team'),
            (1.00, 'Early Lifecycle Retention', 'Account Manager Call + Premium Welcome Benefit', 'Very high-risk early-life customers should receive senior attention.', 'Account Manager' if is_high_value else 'Retention Team'),
        ],
        'bundle': [
            (0.60, 'Bundle Attachment', 'Add-on Discovery Email', 'A soft bundle discovery message is enough at this stage.', 'Retention Team'),
            (0.70, 'Bundle Attachment', 'Personalized Add-on Bundle Offer', 'A moderate bundle offer can increase attachment now.', 'Retention Team'),
            (0.80, 'Bundle Attachment', 'Retention Call + Security/Backup Bundle', 'A direct conversation can position the right sticky add-ons.', 'Retention Team'),
            (0.90, 'Bundle Attachment', 'Retention Call + Premium Bundle Offer', 'Higher urgency justifies a stronger bundle save offer.', 'Retention Team'),
            (1.00, 'Bundle Attachment', 'Account Manager Call + Premium Bundle Offer', 'Very high-risk high-value bundle cases deserve senior ownership.', 'Account Manager' if is_high_value else 'Retention Team'),
        ],
        'general': [
            (0.60, 'General Retention', 'SMS Discount Offer', 'Risk is now high enough for a light save offer.', 'Retention Team'),
            (0.70, 'General Retention', 'Nurture Campaign + Check-in Call', 'The customer needs a more active retention touch.', 'Retention Team'),
            (0.80, 'General Retention', 'Retention Call + 10% Discount Offer', 'A direct call plus a moderate offer is appropriate now.', 'Retention Team'),
            (0.90, 'General Retention', 'Immediate Retention Call + 15% Discount Offer', 'Strong urgency justifies the strongest standard save offer.', 'Retention Team'),
            (1.00, 'General Retention', 'Account Manager Call + 15% Discount Offer', 'Very high risk warrants senior-led retention.', 'Account Manager' if is_high_value else 'Retention Team'),
        ],
    }

    if probability <= 0.50:
        for upper_bound, campaign, action, reason, owner in category_actions['low_risk'].get(category, category_actions['low_risk']['general']):
            if probability <= upper_bound:
                return {
                    'campaign': campaign,
                    'recommended_action': action,
                    'reason': reason,
                    'owner': owner,
                    'category': category,
                }

    for upper_bound, campaign, action, reason, owner in category_actions.get(category, category_actions['general']):
        if probability <= upper_bound:
            return {
                'campaign': campaign,
                'recommended_action': action,
                'reason': reason,
                'owner': owner,
                'category': category,
            }

    return {
        'campaign': 'General Retention',
        'recommended_action': 'Immediate Retention Call + 15% Discount Offer',
        'reason': 'Fallback to the strongest standard retention path.',
        'owner': 'Retention Team',
        'category': 'general',
    }


def get_business_churn_reasons(row: pd.Series, top_drivers=None) -> list[dict]:
    risk_label = str(row.get('predicted_risk') or 'LOW RISK').strip().upper() or 'LOW RISK'
    probability = _safe_float(row.get('predicted_probability'))
    monthly_charges = _safe_float(row.get('MonthlyCharges'))
    data_usage = _safe_float(row.get('Data_Usage_GB'))
    tenure = _safe_int(row.get('tenure'))
    support_tickets = _safe_int(row.get('Support_Tickets'))
    app_logins = _safe_int(row.get('App_Logins'))
    contract = str(row.get('Contract') or '').strip().lower()
    online_security = str(row.get('OnlineSecurity') or '').strip().lower()
    online_backup = str(row.get('OnlineBackup') or '').strip().lower()
    internet_service = str(row.get('InternetService') or '').strip().lower()
    payment_method = str(row.get('PaymentMethod') or '').strip().lower()
    impact_map = _get_driver_impact_map(top_drivers or row.get('top_drivers'))

    reason_candidates = []

    def add_reason(label: str, explanation: str, trigger_score: float, feature_keys: list[str], category: str):
        driver_bonus = sum(impact_map.get(feature, 0.0) for feature in feature_keys)
        reason_candidates.append(
            {
                'label': label,
                'explanation': explanation,
                'score': trigger_score + driver_bonus,
                'category': category,
            }
        )

    if contract == 'month-to-month':
        add_reason(
            'Month-to-month contract keeps switching cost low',
            'Because the customer is still on a month-to-month contract, they can exit without a long commitment barrier, which makes retention more fragile.',
            3.0,
            ['Contract'],
            'contract',
        )
    elif contract == 'one year' and probability >= 0.6:
        add_reason(
            'One-year contract provides only moderate retention protection',
            'The customer has some commitment, but not the stronger stability usually seen with longer-term contracts.',
            1.6,
            ['Contract'],
            'contract',
        )

    if support_tickets >= 4:
        add_reason(
            'Repeated support issues may be damaging trust',
            'The high number of support tickets suggests ongoing service pain points or unresolved problems that can directly weaken customer satisfaction.',
            3.1,
            ['Support_Tickets'],
            'support',
        )
    elif support_tickets >= 2:
        add_reason(
            'Service friction is starting to build',
            'Multiple support interactions suggest the customer experience may not be fully smooth, which can gradually increase churn risk.',
            1.8,
            ['Support_Tickets'],
            'support',
        )

    if app_logins <= 2:
        add_reason(
            'Very low digital engagement suggests detachment',
            'Very limited app activity suggests the customer is no longer interacting much with the service, which can be an important warning sign before churn.',
            3.0,
            ['App_Logins'],
            'engagement',
        )
    elif app_logins <= 5:
        add_reason(
            'Low digital engagement may reflect weakening attachment',
            'App usage is lower than expected, which can indicate the service is becoming less central to the customer routine.',
            1.9,
            ['App_Logins'],
            'engagement',
        )

    if monthly_charges >= 90:
        add_reason(
            'High monthly bill may create price pressure',
            'A high monthly bill can make the customer more sensitive to perceived value and more likely to react to cheaper competitor offers.',
            2.8,
            ['MonthlyCharges'],
            'billing',
        )
    elif monthly_charges >= 75 and probability >= 0.55:
        add_reason(
            'Pricing may influence the retention decision',
            'The monthly charge is significant enough that price-value perception could influence whether the customer stays.',
            1.7,
            ['MonthlyCharges'],
            'billing',
        )

    if data_usage >= 20:
        add_reason(
            'Heavy data usage may need a better-fit bundle',
            'The customer is consuming a high amount of data, so the current plan may feel expensive or restrictive compared with a better-fit bundle.',
            3.4,
            ['Data_Usage_GB', 'MonthlyCharges'],
            'data_usage',
        )
    elif data_usage >= 14 and probability >= 0.55:
        add_reason(
            'Growing data needs could outpace the current plan',
            'Usage is elevated enough that a more generous data offer may improve value perception and reduce churn risk.',
            2.2,
            ['Data_Usage_GB'],
            'data_usage',
        )

    if tenure <= 6:
        add_reason(
            'Early-stage customers are easier to lose',
            'Because the customer is still early in the relationship, loyalty habits may not be fully established yet, which makes early churn more likely.',
            2.4,
            ['tenure'],
            'tenure',
        )
    elif tenure <= 12 and probability >= 0.55:
        add_reason(
            'The relationship is still not deeply established',
            'The customer has not been with the company for very long, so retention can still be sensitive to a few negative experiences.',
            1.5,
            ['tenure'],
            'tenure',
        )

    if internet_service != 'none' and online_security == 'no' and online_backup == 'no':
        add_reason(
            'Limited add-on adoption reduces bundle attachment',
            'The customer is not using protective add-ons like online security or backup, which can reduce how embedded the service feels in daily use.',
            1.9,
            ['OnlineSecurity', 'OnlineBackup'],
            'bundle',
        )

    if payment_method == 'electronic check' and probability >= 0.6:
        add_reason(
            'Payment experience may be adding friction',
            'The current payment method often appears in weaker-retention profiles, so a smoother billing setup could improve stickiness.',
            1.4,
            ['PaymentMethod'],
            'billing',
        )

    if risk_label == 'LOW RISK':
        if probability <= 0.20:
            add_reason(
                'Customer currently looks stable',
                'Current signals remain light, so this case is better handled with routine follow-up than with a retention intervention.',
                0.5,
                [],
                'general',
            )
        else:
            add_reason(
                'Only light watch-out signals are visible',
                'The customer is still in the low-risk zone, so a light-touch follow-up or simple monitoring is more appropriate than a targeted retention campaign.',
                0.5,
                [],
                'general',
            )

    if not reason_candidates:
        add_reason(
            'No single business factor stands out strongly',
            'The customer does not currently show one dominant business signal that clearly points to churn on its own.',
            0.5,
            [],
            'general',
        )

    deduped = {}
    for reason in reason_candidates:
        label = reason['label']
        if label not in deduped or reason['score'] > deduped[label]['score']:
            deduped[label] = reason

    return sorted(deduped.values(), key=lambda item: item['score'], reverse=True)[:3]


def build_churn_reason_summary(row: pd.Series, top_drivers=None) -> str:
    reasons = get_business_churn_reasons(row, top_drivers=top_drivers)
    risk_label = str(row.get('predicted_risk') or '').strip().upper()
    if not reasons:
        return 'No clear churn explanation is available yet.'
    if reasons[0]['label'] in {'No single business factor stands out strongly', 'Customer currently looks stable'}:
        return reasons[0]['explanation']

    labels = [reason['label'].lower() for reason in reasons[:3]]
    if len(labels) == 1:
        joined = labels[0]
    elif len(labels) == 2:
        joined = f'{labels[0]} and {labels[1]}'
    else:
        joined = f'{labels[0]}, {labels[1]}, and {labels[2]}'
    if risk_label == 'HIGH RISK':
        return f'This customer is mainly at risk because of {joined}.'
    if risk_label == 'MEDIUM RISK':
        return f'This customer shows meaningful churn warning signals linked to {joined}.'
    return f'This customer is currently lower risk, but the main watch-out factors are {joined}.'


def _build_combined_action_strategy(
    reasons: list[dict],
    probability: float,
    risk_label: str,
    value_tier: str,
) -> dict | None:
    if probability < 0.65 or len(reasons) < 2:
        return None

    primary_reason = reasons[0]
    secondary_reason = reasons[1]
    primary_score = _safe_float(primary_reason.get('score'))
    secondary_score = _safe_float(secondary_reason.get('score'))
    if secondary_score <= 0 or secondary_score < primary_score * 0.65:
        return None

    primary_category = primary_reason.get('category', 'general')
    secondary_category = secondary_reason.get('category', 'general')
    pair = frozenset({primary_category, secondary_category})
    is_high_value = value_tier == 'High value'

    combo_reason = (
        f"The two strongest churn signals are {primary_reason.get('label', 'the primary risk').lower()} "
        f"and {secondary_reason.get('label', 'the secondary risk').lower()}."
    )

    combo_map = {
        frozenset({'engagement', 'contract'}): {
            'campaign': 'Reactivation + Contract Save',
            'recommended_action': (
                'Account Manager Call + Premium Contract Upgrade Offer'
                if is_high_value and probability >= 0.90
                else 'Retention Call + Premium Contract Upgrade Offer'
                if probability >= 0.80
                else 'Retention Call + Contract Upgrade Offer'
            ),
            'reason': f"{combo_reason} Rebuild usage habits while moving the customer to a stickier plan.",
            'owner': 'Account Manager' if is_high_value and probability >= 0.90 else 'Retention Team',
            'category': 'contract',
            'offer_detail': 'usage coaching plus premium contract conversion',
        },
        frozenset({'engagement', 'billing'}): {
            'campaign': 'Reactivation + Price Save',
            'recommended_action': (
                'Immediate Retention Call + 15% Discount Offer'
                if probability >= 0.85
                else 'Retention Call + 10% Discount Offer'
            ),
            'reason': f"{combo_reason} The save path should reconnect the customer and improve value perception together.",
            'owner': 'Retention Team',
            'category': 'billing',
            'offer_detail': 'reactivation support plus 10% to 15% price relief',
        },
        frozenset({'support', 'billing'}): {
            'campaign': 'Service Recovery + Value Save',
            'recommended_action': (
                'Account Manager + Service Recovery Escalation'
                if is_high_value and probability >= 0.90
                else 'Retention Call + Service Recovery'
            ),
            'reason': f"{combo_reason} Resolve the service pain first, then protect value with a stronger save conversation.",
            'owner': 'Account Manager' if is_high_value and probability >= 0.90 else 'Service Recovery Team',
            'category': 'support',
            'offer_detail': 'priority recovery callback with value-save follow-up',
        },
        frozenset({'data_usage', 'contract'}): {
            'campaign': 'Bundle Fit + Contract Save',
            'recommended_action': (
                'Account Manager Call + Premium Contract Upgrade Offer'
                if is_high_value and probability >= 0.90
                else 'Retention Call + Premium Contract Upgrade Offer'
                if probability >= 0.82
                else 'Retention Call + Contract Upgrade Offer'
            ),
            'reason': f"{combo_reason} Pair a better-fit plan conversation with a longer commitment offer.",
            'owner': 'Account Manager' if is_high_value and probability >= 0.90 else 'Retention Team',
            'category': 'contract',
            'offer_detail': 'better-fit plan review plus premium contract conversion',
        },
        frozenset({'engagement', 'bundle'}): {
            'campaign': 'Reactivation + Bundle Attachment',
            'recommended_action': (
                'Account Manager Call + Premium Bundle Offer'
                if is_high_value and probability >= 0.90
                else 'Retention Call + Premium Bundle Offer'
                if probability >= 0.82
                else 'Retention Call + Usage Coaching'
            ),
            'reason': f"{combo_reason} Rebuild day-to-day usage while adding stickier services the customer may value.",
            'owner': 'Account Manager' if is_high_value and probability >= 0.90 else 'Retention Team',
            'category': 'bundle',
            'offer_detail': 'usage coaching plus sticky add-on bundle',
        },
    }

    combo_strategy = combo_map.get(pair)
    if not combo_strategy:
        return None
    if pair == frozenset({'data_usage', 'contract'}) and probability < 0.88:
        return None
    if risk_label == 'LOW RISK':
        return None
    return combo_strategy


def build_customer_retention_playbook(row: pd.Series, top_drivers=None) -> dict:
    driver_story = build_customer_driver_story(row, top_drivers=top_drivers)
    reasons = driver_story['reasons']
    primary_reason = reasons[0] if reasons else {'label': 'No clear reason', 'category': 'general', 'explanation': ''}
    secondary_reason = reasons[1] if len(reasons) > 1 else None
    risk_label = str(row.get('predicted_risk') or 'LOW RISK').strip().upper() or 'LOW RISK'
    probability = _safe_float(row.get('predicted_probability'))
    value_tier, _ = get_customer_value_tier(row)
    primary_category = primary_reason.get('category', 'general')
    if risk_label == 'LOW RISK':
        low_risk_allowed_categories = {'engagement', 'support', 'contract', 'tenure', 'bundle'}
        if probability <= 0.10 or primary_category not in low_risk_allowed_categories:
            primary_category = 'general'
    strategy = get_probability_decile_action(probability, primary_category, value_tier=value_tier)
    combo_strategy = _build_combined_action_strategy(reasons, probability, risk_label, value_tier)
    if combo_strategy:
        strategy = combo_strategy
    template = get_playbook_template(strategy.get('category', primary_category))
    strategy_category = strategy.get('category', primary_category)
    offer_detail = strategy.get('offer_detail', '')
    if not offer_detail:
        if strategy_category == 'data_usage':
            if probability <= 0.60:
                offer_detail = 'Bronze pack'
            elif probability <= 0.80:
                offer_detail = 'Silver pack'
            else:
                offer_detail = 'Gold pack'
        elif strategy_category == 'billing':
            offer_detail = '10% to 15% price relief'
        elif strategy_category == 'support':
            offer_detail = 'priority recovery callback'
        elif strategy_category == 'contract':
            offer_detail = '12- or 24-month renewal upgrade'

    if not combo_strategy and strategy_category == 'data_usage':
        action_note = (
            'This action fits because the customer may need a bigger or better-matched plan, '
            'which can increase the chance of leaving.'
        )
    elif combo_strategy and secondary_reason:
        action_note = (
            f"This action fits because {primary_reason.get('label', 'the primary churn pattern').lower()} "
            f"and {secondary_reason.get('label', 'the secondary churn pattern').lower()} are both driving risk."
        )
    else:
        action_note = (
            f"This action fits because {primary_reason.get('label', 'the current churn pattern').lower()}."
        )
    coaching_note = (
        f"Top customer-specific drivers: {', '.join(driver_story['top_driver_labels'])}."
        if driver_story['top_driver_labels']
        else 'No specific model drivers are available for this customer yet.'
    )

    return {
        'primary_category': strategy_category,
        'primary_reason': primary_reason.get('label', 'No clear reason'),
        'secondary_reason': secondary_reason.get('label', '') if secondary_reason else '',
        'campaign': strategy.get('campaign', template.get('campaign', 'Standard Retention')),
        'recommended_action': normalize_action_label(strategy.get('recommended_action', get_next_best_action(risk_label))),
        'next_step': template.get('next_step', ''),
        'probability_band_label': get_probability_decile_label(probability),
        'offer_detail': offer_detail,
        'reason_summary': strategy.get('reason', primary_reason.get('explanation', '')),
        'execution_owner': strategy.get('owner', 'Retention Team'),
        'watchout_title': driver_story['watchout_title'],
        'watchout_text': driver_story['watchout_text'],
        'top_driver_labels': driver_story['top_driver_labels'],
        'action_note': action_note,
        'coaching_note': coaching_note,
    }


def get_reason_section_title(risk_label: str) -> str:
    normalized = str(risk_label or '').strip().upper()
    if normalized == 'HIGH RISK':
        return 'Why This Customer May Leave'
    if normalized == 'MEDIUM RISK':
        return 'Main Churn Warning Signals'
    return 'Main Watch-Out Factors'


def get_risk_tier_icon(risk_label: str) -> str:
    normalized = str(risk_label or '').strip().upper()
    if normalized == 'HIGH RISK':
        return '🔴'
    if normalized == 'MEDIUM RISK':
        return '🟠'
    return '🟢'


def _build_reason_driver_mix_html(reasons: list[dict]) -> str:
    if len(reasons) < 2:
        return ''
    primary_reason = str(reasons[0].get('label') or '').strip()
    secondary_reason = str(reasons[1].get('label') or '').strip()
    if not primary_reason or not secondary_reason:
        return ''
    return (
        '<span class="reason-summary-driver-label">Driver mix</span>'
        '<div class="driver-chip-row">'
        f'<span class="driver-chip">{primary_reason}</span>'
        f'<span class="driver-chip">{secondary_reason}</span>'
        '</div>'
    )


def render_reason_insight_block(section_title: str, summary_text: str, reasons: list[dict]):
    driver_mix = _build_reason_driver_mix_html(reasons)
    st.markdown(f'### {section_title}')
    st.markdown(
        (
            '<div class="reason-summary-card">'
            '<div class="reason-summary-label">Main Takeaway</div>'
            f'<div class="reason-summary-text">{summary_text}</div>{driver_mix}'
            '</div>'
        ),
        unsafe_allow_html=True,
    )
    for reason in reasons:
        st.markdown(
            (
                '<div class="reason-card">'
                f'<div class="reason-card-title">{reason["label"]}</div>'
                f'<div class="reason-card-body">{reason["explanation"]}</div>'
                '</div>'
            ),
            unsafe_allow_html=True,
        )


def render_retention_playbook_block(playbook: dict, title: str = 'Campaign'):
    action_fit = (
        playbook.get('action_note')
        or playbook.get('reason_summary')
        or playbook.get('primary_reason')
        or 'No additional action context is available yet.'
    )
    st.markdown(
        (
            '<div class="playbook-card">'
            '<div class="playbook-kicker">Primary Action</div>'
            f'<div class="playbook-title">{title}: {playbook["campaign"]}</div>'
            f'<div class="playbook-body"><strong>Recommended action:</strong> {playbook["recommended_action"]}<br>'
            f'<strong>Why now:</strong> {playbook["primary_reason"]}.<br>'
            f'<strong>Action fit:</strong> {action_fit}</div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )


def build_customer_action_brief(row: pd.Series, top_drivers=None) -> dict:
    playbook = build_customer_retention_playbook(row, top_drivers=top_drivers)
    guidance = get_manager_action_guidance(row)
    reasons = get_business_churn_reasons(row, top_drivers=top_drivers)
    summary_text = build_churn_reason_summary(row, top_drivers=top_drivers)
    risk_label = str(row.get('predicted_risk') or 'LOW RISK').strip().upper()
    top_driver_labels = playbook.get('top_driver_labels', [])

    if risk_label in {'HIGH RISK', 'MEDIUM RISK'}:
        driver_heading = 'Driver features behind this risk'
        factor_heading = 'Factors to catch early'
    else:
        driver_heading = 'Features to keep watching'
        factor_heading = 'Current watch-out factors'

    return {
        'playbook': playbook,
        'guidance': guidance,
        'summary_text': summary_text,
        'driver_heading': driver_heading,
        'factor_heading': factor_heading,
        'factor_items': reasons,
        'show_driver_features': bool(top_driver_labels),
        'top_driver_labels': top_driver_labels,
    }


def build_ranked_manager_recommendations(row: pd.Series, limit: int = 3) -> list[dict]:
    playbook = build_customer_retention_playbook(row)
    risk_label = str(row.get('predicted_risk') or 'LOW RISK').strip().upper()
    manager_options = get_manager_action_options(risk_label)
    recommended_action = playbook['recommended_action']
    primary_category = playbook.get('primary_category', 'general')

    category_affinity = {
        'data_usage': ['Retention call with data plan offer', 'Data plan offer by SMS', 'Email follow-up'],
        'billing': ['Urgent retention call', 'Retention call with discount', 'Discount offer by SMS'],
        'support': ['Service recovery escalation', 'Retention call with service recovery', 'Support follow-up'],
        'engagement': ['Retention call with usage coaching', 'Retention call with win-back offer', 'Check-in call'],
        'contract': ['Retention call with plan upgrade', 'Plan upgrade offer', 'Watch contract end date'],
        'tenure': ['Retention call with onboarding offer', 'Check-in call', 'Email follow-up'],
        'bundle': ['Add-on package offer', 'Email follow-up', 'Watch add-on usage'],
        'general': ['Urgent retention call', 'Discount offer by SMS', 'Email follow-up'],
    }
    rationale_map = {
        'data_usage': 'Best fit if the customer may be outgrowing the current plan.',
        'billing': 'Best fit if price pressure is the main churn trigger.',
        'support': 'Best fit if service issues are pushing the customer away.',
        'engagement': 'Best fit if the customer is becoming less active.',
        'contract': 'Best fit if a stickier plan could improve retention.',
        'tenure': 'Best fit if the customer still needs onboarding reassurance.',
        'bundle': 'Best fit if extra services could make the offer more valuable.',
        'general': 'Best fit for the current overall churn pattern.',
    }
    alternative_reason_map = {
        'Urgent retention call': 'Use when a fast direct intervention matters most.',
        'Retention call with discount': 'Use when a stronger save conversation may be needed.',
        'Account manager escalation': 'Use when the case needs senior ownership.',
        'Retention call with data plan offer': 'Use when the customer may need a better-fit plan.',
        'Service recovery escalation': 'Use when the service issue looks severe.',
        'Retention call with plan upgrade': 'Use when a longer-term plan could reduce churn.',
        'Retention call with onboarding offer': 'Use when the customer is still early in the journey.',
        'Add-on package offer': 'Use when extra services could improve stickiness.',
        'Retention call with usage coaching': 'Use when low engagement is a key concern.',
        'Retention call with win-back offer': 'Use when reactivation is the strongest save path.',
        'Discount offer by SMS': 'Use for a lighter price-focused save attempt.',
        'Check-in call': 'Use for a guided but moderate relationship touch.',
        'Email follow-up': 'Use for a light-touch follow-up that keeps the case warm.',
        'Schedule follow-up': 'Use when the signals are still mild and need checking later.',
        'Data plan offer by SMS': 'Use for a lighter plan-fit adjustment.',
        'Support follow-up': 'Use when service friction needs active follow-through.',
        'Service follow-up SMS': 'Use for a softer service reassurance touch.',
        'Retention call with service recovery': 'Use when recovery should happen through a direct call.',
        'Plan upgrade offer': 'Use when a lighter upgrade offer is enough for now.',
        'No action for now': 'Use when the customer still looks stable.',
        'Keep monitoring': 'Use when watching the case is better than intervening now.',
        'Thank-you message': 'Use when appreciation is enough at this stage.',
        'Watch for better data plan fit': 'Use when plan-fit signals are still early.',
        'Watch contract end date': 'Use when timing matters more than acting now.',
        'Watch add-on usage': 'Use when add-on attachment is still only a watch-out signal.',
    }

    affinity_order = category_affinity.get(primary_category, category_affinity['general'])
    scored_options = []
    for index, option in enumerate(manager_options):
        score = 0
        if option == recommended_action:
            score += 100
        if option in affinity_order:
            score += 60 - (affinity_order.index(option) * 10)

        tone = classify_action_tone(option)
        if risk_label == 'HIGH RISK' and tone in {'critical', 'warning'}:
            score += 12
        elif risk_label == 'MEDIUM RISK' and tone in {'warning', 'informational'}:
            score += 10
        elif risk_label == 'LOW RISK' and tone in {'healthy', 'informational'}:
            score += 10

        score -= index
        scored_options.append((score, option))

    ranked = []
    for rank_index, (_, option) in enumerate(sorted(scored_options, key=lambda item: item[0], reverse=True)[:limit], start=1):
        if rank_index == 1:
            why = rationale_map.get(primary_category, rationale_map['general'])
        else:
            why = alternative_reason_map.get(option, 'Good alternative based on the current risk pattern.')
        ranked.append(
            {
                'rank': rank_index,
                'action': option,
                'why': why,
            }
        )
    return ranked


def get_default_manager_action(
    saved_action,
    manager_options: list[str],
    ranked_recommendations: list[dict],
    recommended_action: str,
) -> str:
    if saved_action in manager_options:
        return saved_action

    if ranked_recommendations:
        top_ranked_action = ranked_recommendations[0].get('action')
        if top_ranked_action in manager_options:
            return top_ranked_action

    if recommended_action in manager_options:
        return recommended_action

    return manager_options[0]


def render_top_driver_chips(driver_labels: list[str]):
    if not driver_labels:
        return
    chips = ''.join(f'<span class="driver-chip">{label}</span>' for label in driver_labels)
    st.markdown(f'<div class="driver-chip-row">{chips}</div>', unsafe_allow_html=True)


def render_ranked_recommendations(recommendations: list[dict], highlighted_action: str | None = None):
    if not recommendations:
        return
    cards = []
    for item in recommendations:
        is_primary = item['action'] == highlighted_action
        card_class = 'ranked-rec-card is-primary' if is_primary else 'ranked-rec-card'
        badge_html = '<span class="ranked-rec-badge">Recommended</span>' if is_primary else ''
        cards.append(
            f'<div class="{card_class}">'
            '<div class="ranked-rec-rank-row">'
            f'<div class="ranked-rec-rank">#{item["rank"]}</div>'
            f'{badge_html}'
            '</div>'
            f'<div class="ranked-rec-title">{item["action"]}</div>'
            f'<div class="ranked-rec-body">{item["why"]}</div>'
            '</div>'
        )
    st.markdown('<div class="ranked-rec-grid">' + ''.join(cards) + '</div>', unsafe_allow_html=True)


def render_customer_action_brief(action_brief: dict):
    playbook = action_brief['playbook']
    guidance = action_brief['guidance']
    st.markdown('**Recommended Action**')
    st.markdown(
        (
            '<div class="playbook-card">'
            f'<div class="playbook-kicker">{playbook["campaign"]}</div>'
            f'<div class="playbook-title">{playbook["recommended_action"]}</div>'
            f'<div class="playbook-body"><strong>Owner:</strong> {guidance["execution_owner"]}<br>'
            f'<strong>Why this action fits:</strong> {playbook.get("action_note", playbook.get("reason_summary", ""))}<br>'
            f'<strong>Next step:</strong> {playbook.get("next_step", "")}</div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )
    if playbook.get('offer_detail'):
        st.caption(f"Offer focus: {playbook['offer_detail']}")


def build_playbook_portfolio_frame(logs: pd.DataFrame, risk_filter: tuple[str, ...] = ('HIGH RISK', 'MEDIUM RISK')) -> pd.DataFrame:
    if logs.empty:
        return pd.DataFrame()

    portfolio_rows = []
    filtered_logs = logs[logs['predicted_risk'].isin(risk_filter)].copy()
    for _, row in filtered_logs.iterrows():
        playbook = build_customer_retention_playbook(row)
        revenue_at_risk = _safe_float(row.get('Revenue at Risk'))
        if revenue_at_risk <= 0:
            revenue_at_risk = _safe_float(row.get('MonthlyCharges'))
        portfolio_rows.append(
            {
                'category': playbook['primary_category'],
                'campaign': playbook['campaign'],
                'recommended_action': playbook['recommended_action'],
                'customers': 1,
                'revenue_at_risk': revenue_at_risk,
                'avg_probability': _safe_float(row.get('predicted_probability')),
            }
        )

    if not portfolio_rows:
        return pd.DataFrame()

    portfolio_df = pd.DataFrame(portfolio_rows)
    summary_df = (
        portfolio_df.groupby(['category', 'campaign', 'recommended_action'], dropna=False)
        .agg(
            customers=('customers', 'sum'),
            revenue_at_risk=('revenue_at_risk', 'sum'),
            avg_probability=('avg_probability', 'mean'),
        )
        .reset_index()
        .sort_values(['customers', 'revenue_at_risk', 'avg_probability'], ascending=[False, False, False])
    )
    return summary_df


def build_campaign_trigger_decision(playbook_summary_df: pd.DataFrame) -> dict:
    if playbook_summary_df.empty:
        return {
            'status': 'Wait For More Cases',
            'campaign': 'No campaign yet',
            'reason': 'There are not enough medium- or high-risk cases to justify a campaign launch yet.',
            'action': 'Keep monitoring incoming predictions.',
            'top_row': None,
        }

    top_row = playbook_summary_df.iloc[0]
    customers = int(top_row['customers'])
    revenue_at_risk = _safe_float(top_row['revenue_at_risk'])
    avg_probability = _safe_float(top_row['avg_probability'])

    if customers >= 5 or revenue_at_risk >= 350 or avg_probability >= 0.78:
        status = 'Launch Now'
        reason = (
            f"{customers} customer(s) currently map to this campaign, with "
            f"{format_currency_tnd(revenue_at_risk)} in revenue exposure and {avg_probability:.1%} average churn risk."
        )
    elif customers >= 3 or revenue_at_risk >= 200 or avg_probability >= 0.65:
        status = 'Prepare Campaign'
        reason = (
            f"The signal is building: {customers} customer(s), {format_currency_tnd(revenue_at_risk)} at risk, "
            f"and {avg_probability:.1%} average churn risk."
        )
    else:
        status = 'Monitor'
        reason = (
            f"The leading campaign is visible, but only {customers} customer(s) are in scope so far "
            f"with {format_currency_tnd(revenue_at_risk)} at risk."
        )

    return {
        'status': status,
        'campaign': str(top_row['campaign']),
        'reason': reason,
        'action': str(top_row['recommended_action']),
        'top_row': top_row,
    }


def get_customer_value_tier(row: pd.Series) -> tuple[str, list[str]]:
    monthly_charges = _safe_float(row.get('MonthlyCharges'))
    tenure = _safe_int(row.get('tenure'))
    reasons = []

    if monthly_charges >= 90:
        reasons.append('high monthly revenue')
    elif monthly_charges >= 70:
        reasons.append('solid monthly revenue')

    if tenure >= 36:
        reasons.append('long relationship history')
    elif tenure >= 18:
        reasons.append('established customer history')

    if monthly_charges >= 90 or (monthly_charges >= 70 and tenure >= 24):
        return 'High value', reasons
    if monthly_charges >= 70 or tenure >= 18:
        return 'Medium value', reasons
    return 'Standard value', reasons or ['limited revenue history so far']


def get_customer_severity_tier(row: pd.Series) -> tuple[str, list[str]]:
    probability = _safe_float(row.get('predicted_probability'))
    support_tickets = _safe_int(row.get('Support_Tickets'))
    app_logins = _safe_int(row.get('App_Logins'))
    contract = str(row.get('Contract') or '').strip().lower()
    reasons = []
    severity_score = 0

    if probability >= 0.85:
        severity_score += 2
        reasons.append('very high predicted churn probability')
    elif probability >= 0.79:
        severity_score += 1
        reasons.append('high predicted churn probability')

    if support_tickets >= 4:
        severity_score += 2
        reasons.append('many recent support tickets')
    elif support_tickets >= 2:
        severity_score += 1
        reasons.append('some service friction')

    if app_logins <= 2:
        severity_score += 2
        reasons.append('very low app engagement')
    elif app_logins <= 5:
        severity_score += 1
        reasons.append('low app engagement')

    if contract == 'month-to-month':
        severity_score += 1
        reasons.append('easy-to-leave contract')

    if severity_score >= 5:
        return 'High severity', reasons
    if severity_score >= 3:
        return 'Medium severity', reasons
    return 'Lower severity', reasons or ['fewer visible urgency signals']


def get_manager_action_guidance(row: pd.Series) -> dict:
    risk_label = str(row.get('predicted_risk') or '')
    probability = _safe_float(row.get('predicted_probability'))
    value_tier, value_reasons = get_customer_value_tier(row)
    severity_tier, severity_reasons = get_customer_severity_tier(row)
    playbook = build_customer_retention_playbook(row)
    recommended_action = playbook['recommended_action']
    execution_owner = playbook.get('execution_owner', 'Retention Team')
    probability_band_label = playbook.get('probability_band_label', get_probability_decile_label(probability))
    top_driver_labels = playbook.get('top_driver_labels', [])

    if risk_label == 'LOW RISK':
        rationale = (
            f'This customer sits in the {probability_band_label} probability band, '
            'so the system keeps the action light and relationship-focused around the current watch-out factors.'
        )
    elif risk_label == 'MEDIUM RISK':
        rationale = (
            f'This customer sits in the {probability_band_label} probability band, '
            'so the system uses a moderate action matched to the strongest warning signals.'
        )
    else:
        rationale = (
            f'This customer sits in the {probability_band_label} probability band, '
            'so the system recommends a strong save action matched to the most likely leave drivers.'
        )

    value_summary = ', '.join(value_reasons[:2])
    severity_summary = ', '.join(severity_reasons[:3])

    return {
        'recommended_action': recommended_action,
        'rationale': rationale,
        'value_tier': value_tier,
        'value_summary': value_summary,
        'severity_tier': severity_tier,
        'severity_summary': severity_summary,
        'probability_band': f'{probability:.1%}',
        'probability_band_label': probability_band_label,
        'execution_owner': execution_owner,
        'playbook_campaign': playbook['campaign'],
        'playbook_category': playbook['primary_category'],
        'playbook_next_step': playbook['next_step'],
        'playbook_offer_detail': playbook['offer_detail'],
        'top_driver_labels': top_driver_labels,
        'watchout_title': playbook.get('watchout_title', get_reason_section_title(risk_label)),
        'watchout_text': playbook.get('watchout_text', ''),
        'action_note': playbook.get('action_note', ''),
        'coaching_note': playbook.get('coaching_note', ''),
    }


def save_manager_action(connection, prediction_id: int, manager_action: str):
    update_fn = getattr(db_utils, 'update_prediction_manager_action', None)
    if update_fn is not None:
        return update_fn(connection, prediction_id, manager_action)

    with connection.cursor() as cursor:
        cursor.execute(
            '''
            ALTER TABLE prediction_logs
            ADD COLUMN IF NOT EXISTS manager_action TEXT,
            ADD COLUMN IF NOT EXISTS manager_action_at TIMESTAMPTZ;
            '''
        )
        cursor.execute(
            '''
            UPDATE prediction_logs
            SET manager_action = %s,
                manager_action_at = NOW()
            WHERE id = %s;
            ''',
            (manager_action, prediction_id),
        )
    connection.commit()


@st.cache_data(ttl=5, show_spinner=False)
def load_manager_actions() -> pd.DataFrame:
    connection = get_postgres_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                '''
                ALTER TABLE prediction_logs
                ADD COLUMN IF NOT EXISTS manager_action TEXT,
                ADD COLUMN IF NOT EXISTS manager_action_at TIMESTAMPTZ;
                '''
            )
            cursor.execute(
                '''
                SELECT id, manager_action, manager_action_at
                FROM prediction_logs;
                '''
            )
            rows = cursor.fetchall()
        connection.commit()
    finally:
        connection.close()

    manager_actions_df = pd.DataFrame(rows, columns=['id', 'manager_action', 'manager_action_at'])
    if not manager_actions_df.empty and 'manager_action' in manager_actions_df.columns:
        manager_actions_df['manager_action'] = manager_actions_df['manager_action'].apply(
            lambda value: normalize_action_label(value) if pd.notna(value) else value
        )
    return manager_actions_df


def hash_dataframe_for_cache(df: pd.DataFrame) -> str:
    """Build a stable cache key even when object columns contain lists or dicts."""
    normalized = df.copy()
    object_columns = normalized.select_dtypes(include=['object']).columns
    if len(object_columns) > 0:
        normalized[object_columns] = normalized[object_columns].applymap(lambda value: json.dumps(value, sort_keys=True, default=str))
    return normalized.to_json(date_format='iso', orient='split', default_handler=str)


@st.cache_data(ttl=5, show_spinner=False, hash_funcs={pd.DataFrame: hash_dataframe_for_cache})
def build_decision_support_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    df = add_prediction_source_columns(prepare_monitoring_dataframe(predictions))
    if df.empty:
        return df

    manager_actions_df = load_manager_actions()
    if not manager_actions_df.empty:
        df = df.drop(columns=['manager_action', 'manager_action_at'], errors='ignore').merge(
            manager_actions_df,
            on='id',
            how='left',
        )

    df['Next Best Action'] = df['predicted_risk'].map(get_next_best_action)
    playbook_df = df.apply(lambda row: pd.Series(build_customer_retention_playbook(row)), axis=1)
    playbook_df = playbook_df.rename(
        columns={
            'primary_category': 'Primary Driver Category',
            'primary_reason': 'Primary Churn Reason',
            'campaign': 'Recommended Campaign',
            'recommended_action': 'Playbook Action',
            'next_step': 'Recommended Next Step',
            'offer_detail': 'Suggested Offer',
            'reason_summary': 'Playbook Reason Summary',
        }
    )
    df = pd.concat([df, playbook_df], axis=1)
    df['Manager Action'] = df.get('manager_action', pd.Series(index=df.index, dtype='object')).fillna('Not selected')
    df['Action Status'] = np.where(df['Manager Action'].eq('Not selected'), 'Pending Action', 'Action Taken')
    df['Outcome Status'] = np.where(df['actual_label'].isna(), 'Outcome Pending', 'Outcome Known')
    df['MonthlyCharges'] = pd.to_numeric(df.get('MonthlyCharges'), errors='coerce').fillna(0.0)
    df['Revenue at Risk'] = np.where(df['predicted_risk'].eq('HIGH RISK'), df['MonthlyCharges'], 0.0)
    return df


def get_manager_recommended_action(alerts: list[dict], high_risk_count: int, monitoring_df: pd.DataFrame | None = None) -> str:
    alert_types = {alert.get('alert_type') for alert in alerts}
    has_high_severity = any(alert.get('severity') == 'high' for alert in alerts)

    if has_high_severity or 'probability_drift' in alert_types or 'live_f1_drop' in alert_types:
        return 'Hold - Model Drift Detected'
    if monitoring_df is not None and not monitoring_df.empty:
        playbook_summary = build_playbook_portfolio_frame(monitoring_df)
        if not playbook_summary.empty:
            top_campaign = playbook_summary.iloc[0]['campaign']
            return str(top_campaign)
    if high_risk_count > 0:
        return 'Launch 10% Discount Campaign'
    return 'Maintain Current Retention Plan'


# --- Demo Data Functions ---
def build_demo_customer_profiles() -> list[dict]:
    return [
        {
            'gender': 'Female', 'SeniorCitizen': 'No', 'tenure': 62, 'MonthlyCharges': 58.0,
            'Contract': 'two year', 'Data_Usage_GB': 8.0, 'Support_Tickets': 0, 'App_Logins': 18,
            'PaymentMethod': 'Bank Transfer (automatic)', 'InternetService': 'DSL', 'PaperlessBilling': 'No',
            'Gouvernorat': 'Tunis', 'Dependents': 'Yes', 'OnlineSecurity': 'Yes', 'OnlineBackup': 'Yes',
            'DeviceProtection': 'Yes', 'StreamingTV': 'No', 'StreamingMovies': 'No',
        },
        {
            'gender': 'Male', 'SeniorCitizen': 'No', 'tenure': 48, 'MonthlyCharges': 64.0,
            'Contract': 'one year', 'Data_Usage_GB': 11.0, 'Support_Tickets': 1, 'App_Logins': 14,
            'PaymentMethod': 'Credit Card (automatic)', 'InternetService': 'DSL', 'PaperlessBilling': 'No',
            'Gouvernorat': 'Sfax', 'Dependents': 'Yes', 'OnlineSecurity': 'Yes', 'OnlineBackup': 'Yes',
            'DeviceProtection': 'No', 'StreamingTV': 'No', 'StreamingMovies': 'No',
        },
        {
            'gender': 'Female', 'SeniorCitizen': 'No', 'tenure': 72, 'MonthlyCharges': 72.0,
            'Contract': 'two year', 'Data_Usage_GB': 15.0, 'Support_Tickets': 0, 'App_Logins': 20,
            'PaymentMethod': 'Bank Transfer (automatic)', 'InternetService': 'Fiber optic', 'PaperlessBilling': 'Yes',
            'Gouvernorat': 'Sousse', 'Dependents': 'Yes', 'OnlineSecurity': 'Yes', 'OnlineBackup': 'Yes',
            'DeviceProtection': 'Yes', 'StreamingTV': 'Yes', 'StreamingMovies': 'Yes',
        },
        {
            'gender': 'Male', 'SeniorCitizen': 'No', 'tenure': 18, 'MonthlyCharges': 82.0,
            'Contract': 'month-to-month', 'Data_Usage_GB': 18.0, 'Support_Tickets': 2, 'App_Logins': 7,
            'PaymentMethod': 'Electronic Check', 'InternetService': 'Fiber optic', 'PaperlessBilling': 'Yes',
            'Gouvernorat': 'Tunis', 'Dependents': 'No', 'OnlineSecurity': 'No', 'OnlineBackup': 'Yes',
            'DeviceProtection': 'No', 'StreamingTV': 'Yes', 'StreamingMovies': 'Yes',
        },
        {
            'gender': 'Female', 'SeniorCitizen': 'Yes', 'tenure': 14, 'MonthlyCharges': 91.0,
            'Contract': 'month-to-month', 'Data_Usage_GB': 14.0, 'Support_Tickets': 3, 'App_Logins': 4,
            'PaymentMethod': 'Electronic Check', 'InternetService': 'Fiber optic', 'PaperlessBilling': 'Yes',
            'Gouvernorat': 'Nabeul', 'Dependents': 'No', 'OnlineSecurity': 'No', 'OnlineBackup': 'No',
            'DeviceProtection': 'No', 'StreamingTV': 'Yes', 'StreamingMovies': 'Yes',
        },
        {
            'gender': 'Male', 'SeniorCitizen': 'No', 'tenure': 9, 'MonthlyCharges': 86.0,
            'Contract': 'month-to-month', 'Data_Usage_GB': 21.0, 'Support_Tickets': 4, 'App_Logins': 3,
            'PaymentMethod': 'Electronic Check', 'InternetService': 'Fiber optic', 'PaperlessBilling': 'Yes',
            'Gouvernorat': 'Monastir', 'Dependents': 'No', 'OnlineSecurity': 'No', 'OnlineBackup': 'No',
            'DeviceProtection': 'No', 'StreamingTV': 'No', 'StreamingMovies': 'Yes',
        },
        {
            'gender': 'Female', 'SeniorCitizen': 'Yes', 'tenure': 4, 'MonthlyCharges': 104.0,
            'Contract': 'month-to-month', 'Data_Usage_GB': 24.0, 'Support_Tickets': 5, 'App_Logins': 1,
            'PaymentMethod': 'Electronic Check', 'InternetService': 'Fiber optic', 'PaperlessBilling': 'Yes',
            'Gouvernorat': 'Tunis', 'Dependents': 'No', 'OnlineSecurity': 'No', 'OnlineBackup': 'No',
            'DeviceProtection': 'No', 'StreamingTV': 'Yes', 'StreamingMovies': 'Yes',
        },
        {
            'gender': 'Male', 'SeniorCitizen': 'No', 'tenure': 27, 'MonthlyCharges': 77.0,
            'Contract': 'one year', 'Data_Usage_GB': 12.0, 'Support_Tickets': 1, 'App_Logins': 10,
            'PaymentMethod': 'Credit Card (automatic)', 'InternetService': 'DSL', 'PaperlessBilling': 'Yes',
            'Gouvernorat': 'Ariana', 'Dependents': 'No', 'OnlineSecurity': 'Yes', 'OnlineBackup': 'No',
            'DeviceProtection': 'Yes', 'StreamingTV': 'No', 'StreamingMovies': 'No',
        },
        {
            'gender': 'Female', 'SeniorCitizen': 'No', 'tenure': 33, 'MonthlyCharges': 69.0,
            'Contract': 'one year', 'Data_Usage_GB': 10.0, 'Support_Tickets': 1, 'App_Logins': 12,
            'PaymentMethod': 'Mailed Check', 'InternetService': 'DSL', 'PaperlessBilling': 'No',
            'Gouvernorat': 'Ben Arous', 'Dependents': 'Yes', 'OnlineSecurity': 'Yes', 'OnlineBackup': 'Yes',
            'DeviceProtection': 'No', 'StreamingTV': 'No', 'StreamingMovies': 'No',
        },
    ]


def seed_demo_predictions(connection, model, dv, scaler) -> int:
    bundle_metadata = get_bundle_metadata()
    decision_threshold = float(bundle_metadata.get('decision_threshold', DEFAULT_DECISION_THRESHOLD))
    inserted_count = 0
    for customer_data in build_demo_customer_profiles():
        X_scaled, _ = prepare_customer_features(customer_data, dv, scaler)
        probability = float(model.predict_proba(X_scaled)[0, 1])
        risk_label, _ = get_risk_label(probability)
        predicted_label = classify_from_probability(probability, threshold=decision_threshold)
        top_drivers = serialize_top_drivers(explain_top_drivers(X_scaled, model, dv))
        insert_prediction_log(
            connection,
            input_features=customer_data,
            predicted_probability=probability,
            predicted_label=predicted_label,
            predicted_risk=risk_label,
            top_drivers=top_drivers,
            model_name=str(bundle_metadata.get('model_family', 'logistic_regression')),
            model_version=str(bundle_metadata.get('variant_name', 'logistic_regression')),
            model_stage='Production',
            source='demo_seed',
            mlflow_run_id=bundle_metadata.get('mlflow_run_id') or bundle_metadata.get('run_id'),
        )
        inserted_count += 1
    return inserted_count


def seed_demo_ground_truth(connection, monitoring_df: pd.DataFrame) -> int:
    unlabeled = monitoring_df[monitoring_df['actual_label'].isna()].copy().sort_values('created_at')
    if unlabeled.empty:
        return 0

    updated_count = 0
    for index, row in unlabeled.head(24).iterrows():
        risk_label = row['predicted_risk']
        if risk_label == 'HIGH RISK':
            actual_label = 'Churn' if updated_count % 4 != 0 else 'No churn'
        elif risk_label == 'MEDIUM RISK':
            actual_label = 'Churn' if updated_count % 2 == 0 else 'No churn'
        else:
            actual_label = 'No churn' if updated_count % 5 != 0 else 'Churn'

        update_prediction_ground_truth(
            connection,
            prediction_id=int(row['id']),
            actual_label=actual_label,
            ground_truth_source='demo_seed',
            feedback_notes='Demo seeded outcome for balanced monitoring behavior.',
        )
        updated_count += 1

    return updated_count


# --- Styling Functions ---
def style_next_best_action(value: str) -> str:
    tone = classify_action_tone(value)
    if tone == 'critical':
        return 'background-color: #fde2e4; color: #9d0208; font-weight: 700;'
    if tone == 'warning':
        return 'background-color: #fff3cd; color: #8d6e00; font-weight: 700;'
    if tone == 'informational':
        return 'background-color: #e8f4f8; color: #0f62a8; font-weight: 600;'
    if tone == 'healthy':
        return 'background-color: #d8f3dc; color: #1b4332; font-weight: 700;'
    if tone == 'neutral':
        return 'background-color: #f1f3f5; color: #495057; font-weight: 600;'
    return ''


def get_action_banner_class(action: str) -> str:
    if action == 'Hold - Model Drift Detected':
        return 'critical'
    tone = classify_action_tone(action)
    if tone == 'critical':
        return 'critical'
    if tone in {'warning', 'informational'} or action == 'Launch 10% Discount Campaign':
        return 'warning'
    return 'healthy'


# --- Chart Building Functions ---
def build_metric_comparison_chart(selected_metrics: dict, baseline_metrics: dict):
    comparison_df = pd.DataFrame([
        {'Metric': 'Accuracy', 'Model': 'Production', 'Score': baseline_metrics['accuracy']},
        {'Metric': 'Accuracy', 'Model': 'Candidate', 'Score': selected_metrics['accuracy']},
        {'Metric': 'F1-Score', 'Model': 'Production', 'Score': baseline_metrics['f1']},
        {'Metric': 'F1-Score', 'Model': 'Candidate', 'Score': selected_metrics['f1']},
        {'Metric': 'ROC-AUC', 'Model': 'Production', 'Score': baseline_metrics['roc_auc']},
        {'Metric': 'ROC-AUC', 'Model': 'Candidate', 'Score': selected_metrics['roc_auc']},
    ])
    figure = px.bar(
        comparison_df, x='Metric', y='Score', color='Model', barmode='group',
        text='Score', color_discrete_sequence=['#8da9c4', '#0f62a8'],
    )
    figure.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    figure.update_layout(height=420, legend_title_text='', margin={'l': 20, 'r': 20, 't': 30, 'b': 20})
    return figure


def build_precision_recall_figure(selected_result: dict, baseline_result: dict):
    figure = go.Figure()
    for label, result, color in [('Production', baseline_result, '#8da9c4'), ('Candidate', selected_result, '#0f62a8')]:
        pr_df = result['precision_recall']
        figure.add_trace(go.Scatter(
            x=pr_df['recall'], y=pr_df['precision'], mode='lines',
            name=f"{label} (PR AUC={float(pr_df['pr_auc'].iloc[0]):.3f})",
            line={'width': 3, 'color': color},
        ))
    figure.update_layout(height=420, xaxis_title='Recall', yaxis_title='Precision', legend_title_text='')
    return figure


def build_risk_distribution_chart(logs: pd.DataFrame):
    risk_counts = logs['predicted_risk'].value_counts().rename_axis('risk_level').reset_index(name='count')
    figure = px.pie(
        risk_counts, values='count', names='risk_level', hole=0.60,
        color='risk_level',
        color_discrete_map={'HIGH RISK': '#d62839', 'MEDIUM RISK': '#f4a261', 'LOW RISK': '#2a9d8f'},
    )
    figure.update_traces(textposition='inside', textinfo='percent+label')
    figure.update_layout(height=410, legend_title_text='')
    return figure


def build_probability_drift_chart(drift_df: pd.DataFrame):
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=drift_df['created_at'], y=drift_df['avg_probability'],
        mode='lines+markers', name='Avg probability', line={'width': 3, 'color': '#0f62a8'},
    ))
    figure.add_trace(go.Scatter(
        x=drift_df['created_at'], y=drift_df['high_risk_share'],
        mode='lines+markers', name='High-risk share', line={'width': 3, 'color': '#d62839'}, yaxis='y2',
    ))
    figure.update_layout(
        height=410, xaxis_title='Prediction time', yaxis={'title': 'Avg churn probability'},
        yaxis2={'title': 'High-risk share', 'overlaying': 'y', 'side': 'right', 'tickformat': '.0%'},
        legend_title_text='',
    )
    return figure


def build_reason_distribution_frame(logs: pd.DataFrame, risk_filter: str = 'HIGH RISK') -> pd.DataFrame:
    if logs.empty or 'predicted_risk' not in logs.columns:
        return pd.DataFrame(columns=['reason', 'count'])

    filtered = logs[logs['predicted_risk'].eq(risk_filter)].copy()
    if filtered.empty:
        return pd.DataFrame(columns=['reason', 'count'])

    reason_counts = {}
    for _, row in filtered.iterrows():
        for reason in get_business_churn_reasons(row):
            label = reason['label']
            reason_counts[label] = reason_counts.get(label, 0) + 1

    if not reason_counts:
        return pd.DataFrame(columns=['reason', 'count'])

    return (
        pd.DataFrame(
            [{'reason': reason, 'count': count} for reason, count in reason_counts.items()]
        )
        .sort_values(['count', 'reason'], ascending=[False, True])
        .head(6)
    )


def build_driver_frequency_frame(logs: pd.DataFrame) -> pd.DataFrame:
    if logs.empty or 'top_drivers' not in logs.columns:
        return pd.DataFrame(columns=['feature', 'count'])

    feature_counts = {}
    for _, row in logs.iterrows():
        seen_features = set()
        for driver in row.get('top_drivers', []) or []:
            if not isinstance(driver, dict):
                continue
            feature = str(driver.get('feature') or '').strip()
            if not feature or feature in seen_features:
                continue
            seen_features.add(feature)
            feature_counts[feature] = feature_counts.get(feature, 0) + 1

    if not feature_counts:
        return pd.DataFrame(columns=['feature', 'count'])

    return (
        pd.DataFrame(
            [{'feature': format_feature_name(feature), 'count': count} for feature, count in feature_counts.items()]
        )
        .sort_values(['count', 'feature'], ascending=[False, True])
        .head(8)
    )


def build_lifecycle_figure(lifecycle_df: pd.DataFrame):
    color_map = {
        'Completed': '#2a9d8f', 'Active': '#0f62a8', 'Pending': '#adb5bd',
        'approve_candidate': '#2a9d8f', 'review_candidate': '#f4a261', 'reject_candidate': '#d62839',
        'Healthy': '#2a9d8f', 'Monitor Closely': '#f4a261', 'Retraining Recommended': '#d62839',
    }
    statuses = lifecycle_df['status'].tolist()
    colors = [color_map.get(status, '#0f62a8') for status in statuses]

    figure = go.Figure(data=[go.Scatter(
        x=list(range(len(lifecycle_df))), y=[1] * len(lifecycle_df),
        mode='markers+lines+text', marker={'size': 26, 'color': colors},
        line={'color': '#8da9c4', 'width': 4}, text=lifecycle_df['step'],
        textposition='top center',
        customdata=lifecycle_df[['status', 'description']].values,
        hovertemplate='<b>%{text}</b><br>Status: %{customdata[0]}<br>%{customdata[1]}<extra></extra>',
    )])
    figure.update_layout(
        height=230, margin={'l': 20, 'r': 20, 't': 30, 'b': 20},
        yaxis={'visible': False}, xaxis={'visible': False}, showlegend=False,
    )
    return figure


# --- Analytics Functions ---
def prepare_analytics_dataset(df: pd.DataFrame) -> pd.DataFrame:
    analytics_df = df.copy()
    analytics_df['Churn_Binary'] = (analytics_df['Churn'] == 'Yes').astype(int)
    analytics_df['TenureBucket'] = pd.cut(
        analytics_df['tenure'], bins=[0, 6, 12, 24, 48, 72],
        labels=['0-6', '7-12', '13-24', '25-48', '49+'], include_lowest=True,
    )
    return analytics_df


def build_saved_predictions_analytics_dataset(predictions: pd.DataFrame) -> pd.DataFrame:
    saved_df = build_decision_support_frame(predictions)
    if saved_df.empty:
        return pd.DataFrame()

    analytics_df = saved_df.copy()
    actual_label_series = analytics_df.get('actual_label', pd.Series(index=analytics_df.index, dtype='object'))
    predicted_label_series = analytics_df.get('predicted_label', pd.Series(index=analytics_df.index, dtype='object'))
    resolved_churn = actual_label_series.where(actual_label_series.notna(), predicted_label_series)
    analytics_df['Churn'] = resolved_churn.replace({'Churn': 'Yes', 'No churn': 'No'})
    analytics_df['Data Source'] = 'Saved Predictions'

    service_columns = [
        column for column in ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'StreamingTV', 'StreamingMovies']
        if column in analytics_df.columns
    ]
    if service_columns:
        analytics_df['TotalServices'] = analytics_df[service_columns].apply(
            lambda row: int(sum(str(value).strip().lower() == 'yes' for value in row)),
            axis=1,
        )

    if {'MonthlyCharges', 'tenure'}.issubset(analytics_df.columns):
        analytics_df['Total_Revenue'] = (
            pd.to_numeric(analytics_df['MonthlyCharges'], errors='coerce').fillna(0.0)
            * pd.to_numeric(analytics_df['tenure'], errors='coerce').fillna(0.0)
        )
    if {'Data_Usage_GB', 'tenure'}.issubset(analytics_df.columns):
        safe_tenure = pd.to_numeric(analytics_df['tenure'], errors='coerce').replace(0, np.nan)
        analytics_df['Usage_per_Month'] = (
            pd.to_numeric(analytics_df['Data_Usage_GB'], errors='coerce').fillna(0.0) / safe_tenure
        ).fillna(0.0)

    analytics_df = analytics_df[analytics_df['Churn'].isin(['Yes', 'No'])].copy()
    return analytics_df


def build_churn_rate_chart(df: pd.DataFrame, group_col: str, title: str, color_sequence: list[str]):
    churn_rate_df = (
        df.groupby(group_col, dropna=False)['Churn_Binary'].mean()
        .reset_index(name='churn_rate').sort_values('churn_rate', ascending=False)
    )
    figure = px.bar(churn_rate_df, x=group_col, y='churn_rate', title=title,
                    color=group_col, color_discrete_sequence=color_sequence)
    figure.update_layout(height=360, showlegend=False, yaxis_tickformat='.0%')
    return figure


def build_distribution_chart(df: pd.DataFrame, x_col: str, title: str):
    figure = px.histogram(
        df, x=x_col, color='Churn', barmode='overlay', opacity=0.72, title=title,
        color_discrete_map={'Yes': '#d62839', 'No': '#2a9d8f'},
    )
    figure.update_layout(height=360)
    return figure


def build_geo_churn_chart(df: pd.DataFrame):
    geo_df = (
        df.groupby('Gouvernorat', dropna=False)['Churn_Binary'].mean()
        .reset_index(name='churn_rate').sort_values('churn_rate', ascending=False)
    )
    figure = px.bar(geo_df, x='Gouvernorat', y='churn_rate', color='churn_rate',
                    title='Churn Rate by Governorate', color_continuous_scale=['#2a9d8f', '#f4a261', '#d62839'])
    figure.update_layout(height=380, xaxis_tickangle=-35, yaxis_tickformat='.0%')
    return figure


def build_correlation_heatmap(df: pd.DataFrame):
    correlation_cols = [
        column for column in ['tenure', 'MonthlyCharges', 'Data_Usage_GB', 'TotalServices', 'Total_Revenue', 'Usage_per_Month', 'Churn_Binary']
        if column in df.columns
    ]
    correlation_df = df[correlation_cols].corr(numeric_only=True)
    figure = px.imshow(correlation_df, title='Feature Correlation Matrix',
                       color_continuous_scale=['#2a9d8f', '#f5f8fc', '#d62839'], aspect='auto')
    figure.update_layout(height=430)
    return figure


# --- Analytics Render Functions ---
def render_demography_analysis(df: pd.DataFrame):
    st.markdown('### 🌍 Demography')
    col1, col2 = st.columns(2)
    with col1:
        if 'gender' in df.columns:
            st.plotly_chart(build_churn_rate_chart(df, 'gender', 'Churn Rate by Gender', ['#0f62a8', '#8da9c4']), width='stretch')
        elif 'Gender' in df.columns:
            st.plotly_chart(build_churn_rate_chart(df, 'Gender', 'Churn Rate by Gender', ['#0f62a8', '#8da9c4']), width='stretch')
    with col2:
        senior_col = 'SeniorCitizen' if 'SeniorCitizen' in df.columns else None
        if senior_col:
            senior_df = df.copy()
            senior_df[senior_col] = senior_df[senior_col].replace({1: 'Yes', 0: 'No'})
            st.plotly_chart(build_churn_rate_chart(senior_df, senior_col, 'Churn Rate by Senior Citizen Status', ['#f4a261', '#2a9d8f']), width='stretch')
    if 'Dependents' in df.columns:
        st.plotly_chart(build_churn_rate_chart(df, 'Dependents', 'Churn Rate by Dependents', ['#457b9d', '#a8dadc']), width='stretch')


def render_services_analysis(df: pd.DataFrame):
    st.markdown('### 💳 Services')
    col1, col2 = st.columns(2)
    with col1:
        if 'InternetService' in df.columns:
            st.plotly_chart(build_churn_rate_chart(df, 'InternetService', 'Churn Rate by Internet Service', ['#f4a261', '#e76f51', '#2a9d8f']), width='stretch')
        service_columns = [col for col in ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection'] if col in df.columns]
        if service_columns:
            melted = df.melt(id_vars=['Churn'], value_vars=service_columns, var_name='service', value_name='enabled')
            service_chart = px.histogram(melted, x='service', color='enabled', barmode='group',
                                         title='Adoption of Core Protection Services', color_discrete_map={'Yes': '#0f62a8', 'No': '#d62839'})
            service_chart.update_layout(height=360)
            st.plotly_chart(service_chart, width='stretch')
    with col2:
        stream_columns = [col for col in ['StreamingTV', 'StreamingMovies'] if col in df.columns]
        if stream_columns:
            stream_df = df.copy()
            stream_df['StreamingBundle'] = stream_df[stream_columns].apply(lambda row: 'Enabled' if 'Yes' in row.values else 'Disabled', axis=1)
            st.plotly_chart(build_churn_rate_chart(stream_df, 'StreamingBundle', 'Churn Rate by Streaming Bundle', ['#1d3557', '#a8dadc']), width='stretch')
        if 'PaperlessBilling' in df.columns:
            st.plotly_chart(build_churn_rate_chart(df, 'PaperlessBilling', 'Churn Rate by Paperless Billing', ['#577590', '#90be6d']), width='stretch')


def render_billing_analysis(df: pd.DataFrame):
    st.markdown('### 📡 Bills And Contracts')
    col1, col2 = st.columns(2)
    with col1:
        if 'Contract' in df.columns:
            st.plotly_chart(build_churn_rate_chart(df, 'Contract', 'Churn Rate by Contract Type', ['#0f62a8', '#4ea8de', '#8da9c4']), width='stretch')
        if 'PaymentMethod' in df.columns:
            payment_chart = build_churn_rate_chart(df, 'PaymentMethod', 'Churn Rate by Payment Method', ['#1d3557', '#457b9d', '#a8dadc', '#e9c46a'])
            payment_chart.update_layout(xaxis_tickangle=-25)
            st.plotly_chart(payment_chart, width='stretch')
    with col2:
        if 'MonthlyCharges' in df.columns:
            st.plotly_chart(build_distribution_chart(df, 'MonthlyCharges', 'Monthly Charges Distribution by Churn'), width='stretch')
        if 'MonthlyCharges' in df.columns and 'tenure' in df.columns:
            scatter_plot = px.scatter(df, x='tenure', y='MonthlyCharges', color='Churn', title='Tenure vs Monthly Charges',
                                      color_discrete_map={'Yes': '#d62839', 'No': '#2a9d8f'}, opacity=0.65)
            scatter_plot.update_layout(height=360)
            st.plotly_chart(scatter_plot, width='stretch')


def render_tenure_analysis(df: pd.DataFrame):
    st.markdown('### ⏳ Tenure And Usage')
    col1, col2 = st.columns(2)
    with col1:
        if 'tenure' in df.columns:
            st.plotly_chart(build_distribution_chart(df, 'tenure', 'Tenure Distribution by Churn'), width='stretch')
        tenure_bucket_df = df.groupby('TenureBucket', dropna=False)['Churn_Binary'].mean().reset_index(name='churn_rate')
        tenure_bucket_chart = px.line(tenure_bucket_df, x='TenureBucket', y='churn_rate', markers=True,
                                       title='Churn Rate by Tenure Bucket', color_discrete_sequence=['#0f62a8'])
        tenure_bucket_chart.update_layout(height=360, yaxis_tickformat='.0%')
        st.plotly_chart(tenure_bucket_chart, width='stretch')
    with col2:
        if 'Data_Usage_GB' in df.columns:
            usage_chart = px.histogram(df, x='Data_Usage_GB', color='Churn', barmode='overlay', opacity=0.72,
                                       title='Data Usage Distribution by Churn', color_discrete_map={'Yes': '#d62839', 'No': '#2a9d8f'})
            usage_chart.update_layout(height=360)
            st.plotly_chart(usage_chart, width='stretch')
        if 'Usage_per_Month' in df.columns:
            usage_tenure_chart = px.scatter(df, x='tenure', y='Usage_per_Month', color='Churn', title='Tenure vs Usage Per Month',
                                            color_discrete_map={'Yes': '#d62839', 'No': '#2a9d8f'}, opacity=0.65)
            usage_tenure_chart.update_layout(height=360)
            st.plotly_chart(usage_tenure_chart, width='stretch')


def render_geography_analysis(df: pd.DataFrame):
    st.markdown('### 🗺️ Geography')
    if 'Gouvernorat' in df.columns:
        st.plotly_chart(build_geo_churn_chart(df), width='stretch')


def render_correlation_analysis(df: pd.DataFrame):
    st.markdown('### 🛠️ Feature Correlations')
    st.plotly_chart(build_correlation_heatmap(df), width='stretch')


# ============================================================
# TAB RENDER FUNCTIONS - Managerial Decision Support System
# ============================================================

def render_single_prediction_tab(model, dv, scaler):
    """Single Prediction Tab - Run production model on customer data"""
    st.subheader('📋 Predictions')
    st.caption('Score a customer, explain the churn risk, and record the decision for follow-up.')

    with st.form('single_prediction_form', clear_on_submit=False):
        customer_data = build_customer_inputs('single_prediction')
        submitted = st.form_submit_button('Run Production Prediction', type='primary')

    if not submitted:
        return

    current_model, current_dv, current_scaler = get_cached_model_bundle()
    bundle_metadata = get_bundle_metadata()
    decision_threshold = float(bundle_metadata.get('decision_threshold', DEFAULT_DECISION_THRESHOLD))
    X_scaled, _ = prepare_customer_features(customer_data, current_dv, current_scaler)
    probability = float(current_model.predict_proba(X_scaled)[0, 1])
    risk_label, risk_tag = get_risk_label(probability)
    predicted_label = classify_from_probability(probability, threshold=decision_threshold)
    top_drivers_df = explain_top_drivers(X_scaled, current_model, current_dv)
    top_drivers = serialize_top_drivers(top_drivers_df)
    customer_view = pd.Series({**customer_data, 'predicted_probability': probability, 'predicted_risk': risk_label})
    churn_reasons = get_business_churn_reasons(customer_view, top_drivers=top_drivers)
    churn_summary = build_churn_reason_summary(customer_view, top_drivers=top_drivers)

    st.divider()
    metric_cols = st.columns(3)
    metric_cols[0].metric('Churn Probability', f'{probability:.1%}')
    metric_cols[1].metric('Predicted Label', predicted_label)
    metric_cols[2].metric('Risk Tier', risk_label)
    st.caption(f'Current decision threshold: {decision_threshold:.2f}')

    left_col, right_col = st.columns([0.9, 1.1])
    with left_col:
        st.markdown(f'### {risk_tag}')
        if risk_label == 'HIGH RISK':
            st.error('This customer should be prioritized for retention action.')
        elif risk_label == 'MEDIUM RISK':
            st.warning('This customer deserves closer review and targeted intervention.')
        else:
            st.success('This customer currently appears stable.')
    with right_col:
        render_reason_insight_block(
            get_reason_section_title(risk_label),
            churn_summary,
            churn_reasons,
        )

    with st.expander('Technical driver details'):
        for driver in top_drivers:
            st.write(f"- **{format_feature_name(driver['feature'])}**: {driver['impact']:.3f}")

    connection = get_postgres_connection()
    try:
        bootstrap_platform_tables_if_enabled(connection)
        prediction_id = insert_prediction_log(
            connection, input_features=customer_data, predicted_probability=probability,
            predicted_label=predicted_label, predicted_risk=risk_label, top_drivers=top_drivers,
            model_name=str(bundle_metadata.get('model_family', 'logistic_regression')),
            model_version=str(bundle_metadata.get('variant_name', 'logistic_regression')),
            model_stage='Production', source=FORM_SOURCE,
            mlflow_run_id=bundle_metadata.get('mlflow_run_id') or bundle_metadata.get('run_id'),
        )
        clear_runtime_caches()
        st.success(f'Prediction stored successfully with record ID `{prediction_id}`.')
    except Exception as error:
        st.error(f'Unable to save the prediction to PostgreSQL: {error}')
    finally:
        connection.close()

    with st.expander('Customer Profile Summary'):
        summary_df = pd.DataFrame(
            [{'Field': key, 'Value': value} for key, value in customer_data.items()]
        )
        st.dataframe(summary_df, width='stretch', hide_index=True)


def render_manager_insights_tab(predictions: pd.DataFrame, alerts_df: pd.DataFrame, governance_df: pd.DataFrame):
    """Manager Insights Tab - Executive KPIs and system status"""
    st.subheader('📊 Manager Insights')
    st.caption('Executive view of churn risk, revenue exposure, team actions, and system health.')

    active_bundle_metadata = get_bundle_metadata()
    active_baseline_metrics = get_baseline_metrics_from_metadata(active_bundle_metadata)
    monitoring_df = build_decision_support_frame(
        filter_predictions_for_active_model(predictions, active_bundle_metadata)
    )
    full_history_monitoring_df = build_decision_support_frame(predictions)

    # Calculate KPIs
    if monitoring_df.empty:
        st.info('No prediction data available yet. Run some predictions first.')
        return

    computed_alerts = detect_monitoring_alerts(monitoring_df, active_baseline_metrics)
    kpi_df = full_history_monitoring_df.copy()
    high_risk_count = int((kpi_df['predicted_risk'] == 'HIGH RISK').sum())
    medium_risk_count = int((kpi_df['predicted_risk'] == 'MEDIUM RISK').sum())
    low_risk_count = int((kpi_df['predicted_risk'] == 'LOW RISK').sum())

    avg_monthly_bill = float(kpi_df['MonthlyCharges'].mean()) if 'MonthlyCharges' in kpi_df.columns else 0.0
    revenue_at_risk = calculate_revenue_at_risk_total(kpi_df)
    model_trust_score = calculate_model_trust_score(computed_alerts)
    playbook_summary_df = build_playbook_portfolio_frame(full_history_monitoring_df)
    campaign_trigger = build_campaign_trigger_decision(playbook_summary_df)

    # Top Governorate based on the same revenue-at-risk logic used in the detailed chart
    top_governorate = 'N/A'
    top_governorate_revenue_risk = 0.0
    if 'Gouvernorat' in kpi_df.columns:
        governorate_kpi_df = (
            kpi_df.assign(
                revenue_at_risk=np.where(
                    kpi_df['predicted_risk'].eq('HIGH RISK'),
                    kpi_df['MonthlyCharges'],
                    np.where(kpi_df['predicted_risk'].eq('MEDIUM RISK'), kpi_df['MonthlyCharges'] * 0.5, 0.0),
                )
            )
            .groupby('Gouvernorat', dropna=False)
            .agg(
                revenue_at_risk=('revenue_at_risk', 'sum'),
                avg_risk=('predicted_probability', 'mean'),
            )
            .reset_index()
            .sort_values(['revenue_at_risk', 'avg_risk'], ascending=[False, False])
        )
        if not governorate_kpi_df.empty:
            top_governorate = str(governorate_kpi_df.iloc[0]['Gouvernorat'])
            top_governorate_revenue_risk = float(governorate_kpi_df.iloc[0]['revenue_at_risk'])

    # System Status
    system_status = 'Healthy'
    status_class = 'healthy'
    status_color = '#2a9d8f'
    if any(alert.get('severity') == 'high' for alert in computed_alerts):
        system_status = 'Critical'
        status_class = 'critical'
        status_color = '#d62839'
    elif any(alert.get('severity') == 'medium' for alert in computed_alerts):
        system_status = 'Warning'
        status_class = 'warning'
        status_color = '#f4a261'

    # --- KPI Cards with Containers ---
    render_section_anchor('manager-kpis')
    st.markdown('### Key Performance Indicators')

    kpi_row1 = st.columns(4)
    with kpi_row1[0]:
        with st.container(border=True):
            st.markdown(f'<div class="kpi-value">{format_currency_tnd(revenue_at_risk)}</div>', unsafe_allow_html=True)
            st.markdown('<div class="kpi-label">Revenue at Risk</div>', unsafe_allow_html=True)
    with kpi_row1[1]:
        with st.container(border=True):
            status_html = f'<div class="kpi-value" style="color:{status_color};">{system_status}</div>'
            st.markdown(status_html, unsafe_allow_html=True)
            st.markdown('<div class="kpi-label">System Status</div>', unsafe_allow_html=True)
    with kpi_row1[2]:
        with st.container(border=True):
            st.markdown(f'<div class="kpi-value">{top_governorate}</div>', unsafe_allow_html=True)
            st.markdown('<div class="kpi-label">Top Governorate (Exposure)</div>', unsafe_allow_html=True)
    with kpi_row1[3]:
        with st.container(border=True):
            st.markdown(f'<div class="kpi-value">{model_trust_score}%</div>', unsafe_allow_html=True)
            st.markdown('<div class="kpi-label">Model Trust Score</div>', unsafe_allow_html=True)

    # Risk Distribution Donut Chart
    render_section_anchor('manager-risk-distribution')
    st.markdown('### Risk Distribution')
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.plotly_chart(
            build_risk_distribution_chart(full_history_monitoring_df),
            width='stretch',
            key='monitoring_manager_risk_distribution',
        )
    with col_chart2:
        risk_metric_cols = st.columns(3)
        risk_metric_cols[0].metric('High Risk', high_risk_count)
        risk_metric_cols[1].metric('Medium Risk', medium_risk_count)
        risk_metric_cols[2].metric('Low Risk', low_risk_count)
        st.caption(
            f'Estimated exposure remains concentrated in the high-risk queue: '
            f'{format_currency_tnd(revenue_at_risk)}.'
        )

    render_section_anchor('manager-focus')
    st.markdown('### Where To Focus')
    focus_col1, focus_col2 = st.columns(2)
    with focus_col1:
        if 'Contract' in kpi_df.columns:
            contract_risk = (
                kpi_df.groupby('Contract', dropna=False)['predicted_probability']
                .mean()
                .reset_index()
                .sort_values('predicted_probability', ascending=False)
            )
            contract_risk_chart = px.bar(
                contract_risk,
                x='Contract',
                y='predicted_probability',
                color='predicted_probability',
                title='Average Risk By Contract',
                color_continuous_scale=['#2a9d8f', '#f4a261', '#d62839'],
            )
            contract_risk_chart.update_layout(height=340, coloraxis_showscale=False, yaxis_tickformat='.0%')
            st.plotly_chart(contract_risk_chart, width='stretch', key='manager_contract_risk_chart')
    with focus_col2:
        segment_fields = [field for field in ['Gouvernorat', 'Contract', 'PaymentMethod'] if field in kpi_df.columns]
        if segment_fields and 'MonthlyCharges' in kpi_df.columns:
            segment_priority = (
                kpi_df.groupby(segment_fields, dropna=False)
                .agg(
                    customers=('predicted_probability', 'size'),
                    avg_risk=('predicted_probability', 'mean'),
                    revenue_at_risk=('MonthlyCharges', 'sum'),
                )
                .reset_index()
                .sort_values(['avg_risk', 'revenue_at_risk'], ascending=[False, False])
                .head(8)
            )
            segment_priority['avg_risk'] = segment_priority['avg_risk'].map(lambda value: f'{value:.1%}')
            segment_priority['revenue_at_risk'] = segment_priority['revenue_at_risk'].map(format_currency_tnd)
            st.dataframe(segment_priority, width='stretch', hide_index=True)

    render_section_anchor('manager-revenue-risk')
    st.markdown('### Revenue At Risk By Governorate')
    if {'Gouvernorat', 'MonthlyCharges', 'predicted_risk'}.issubset(kpi_df.columns):
        governorate_risk = (
            kpi_df.assign(
                revenue_at_risk=np.where(
                    kpi_df['predicted_risk'].eq('HIGH RISK'),
                    kpi_df['MonthlyCharges'],
                    np.where(kpi_df['predicted_risk'].eq('MEDIUM RISK'), kpi_df['MonthlyCharges'] * 0.5, 0.0),
                )
            )
            .groupby('Gouvernorat', dropna=False)
            .agg(
                revenue_at_risk=('revenue_at_risk', 'sum'),
                avg_risk=('predicted_probability', 'mean'),
                customers=('predicted_probability', 'size'),
            )
            .reset_index()
            .sort_values('revenue_at_risk', ascending=False)
        )
        geo_col1, geo_col2 = st.columns([1.15, 0.85])
        with geo_col1:
            governorate_risk_chart = px.bar(
                governorate_risk.head(10),
                x='Gouvernorat',
                y='revenue_at_risk',
                color='avg_risk',
                title='Top Governorates By Revenue At Risk',
                color_continuous_scale=['#2a9d8f', '#f4a261', '#d62839'],
            )
            governorate_risk_chart.update_layout(height=360, coloraxis_colorbar_title='Avg risk')
            st.plotly_chart(governorate_risk_chart, width='stretch', key='manager_governorate_revenue_risk_chart')
        with geo_col2:
            if not governorate_risk.empty:
                lead_row = governorate_risk.iloc[0]
                st.metric('Highest-Exposure Governorate', str(lead_row['Gouvernorat']))
                st.metric('Revenue At Risk', format_currency_tnd(lead_row['revenue_at_risk']))
                st.metric('Average Risk', f"{lead_row['avg_risk']:.1%}")
                st.caption('Focus field retention effort where exposure and risk concentration are both highest.')

    # Active Alerts Summary
    if False:
        st.markdown('### Active Alerts')
    if False:
        st.success('✅ No active alerts - system is operating normally.')
    elif False:
        alert_cols = st.columns(3)
        high_alerts = [a for a in computed_alerts if a.get('severity') == 'high']
        medium_alerts = [a for a in computed_alerts if a.get('severity') == 'medium']
        alert_cols[0].metric('🔴 High Severity', len(high_alerts))
        alert_cols[1].metric('🟡 Medium Severity', len(medium_alerts))
        alert_cols[2].metric('📊 Total Alerts', len(computed_alerts))

        with st.expander('View Alert Details'):
            for alert in computed_alerts:
                severity_icon = '🔴' if alert.get('severity') == 'high' else '🟡'
                st.write(f"{severity_icon} {format_monitoring_alert(alert)}")

    # Live Performance Metrics removed from Manager Insights
    pass

    if False:
        if not computed_alerts:
            st.write('')
        else:
            alert_cols = st.columns(3)
            high_alerts = [a for a in computed_alerts if a.get('severity') == 'high']
            medium_alerts = [a for a in computed_alerts if a.get('severity') == 'medium']
            alert_cols[0].metric('High Severity', len(high_alerts))
            alert_cols[1].metric('Medium Severity', len(medium_alerts))
            alert_cols[2].metric('Total Alerts', len(computed_alerts))

            with st.expander('View Alert Details'):
                for alert in computed_alerts:
                    st.write(format_monitoring_alert(alert))

        perf_cols = st.columns(4)
        perf_cols[0].metric('Total Inferences', f'{len(monitoring_df)}')
        perf_cols[1].metric('Avg Churn Probability', f"{monitoring_df['predicted_probability'].mean():.1%}")
        perf_cols[2].metric('Ground-Truth Coverage', '0.0%')
        perf_cols[3].metric('Live F1-Score', 'N/A')

    st.divider()
    st.markdown('### Why Churn Risk Is Rising')
    analytics_col1, analytics_col2 = st.columns(2)
    with analytics_col1:
        reason_distribution_df = build_reason_distribution_frame(full_history_monitoring_df, risk_filter='HIGH RISK')
        if reason_distribution_df.empty:
            st.info('Add more high-risk predictions to surface the main business reasons behind churn risk.')
        else:
            lead_reason = reason_distribution_df.iloc[0]
            st.caption(
                f"Most common high-risk reason: `{lead_reason['reason']}` appears in {int(lead_reason['count'])} saved case(s)."
            )
            reason_chart = px.bar(
                reason_distribution_df.sort_values('count', ascending=True),
                x='count',
                y='reason',
                orientation='h',
                title='Top Business Reasons Among High-Risk Customers',
                color='count',
                color_continuous_scale=['#dceefb', '#74a9cf', '#045a8d'],
            )
            reason_chart.update_layout(
                height=360,
                xaxis_title='Customers',
                yaxis_title='Business reason',
                coloraxis_showscale=False,
            )
            st.plotly_chart(reason_chart, width='stretch', key='manager_top_business_reasons_chart')
    with analytics_col2:
        driver_frequency_df = build_driver_frequency_frame(full_history_monitoring_df)
        if driver_frequency_df.empty:
            st.info('Driver frequency will appear after more saved predictions accumulate.')
        else:
            top_feature = driver_frequency_df.iloc[0]
            st.caption(
                f"Most frequent model signal: `{top_feature['feature']}` appears in {int(top_feature['count'])} saved explanation set(s)."
            )
            driver_chart = px.bar(
                driver_frequency_df.sort_values('count', ascending=True),
                x='count',
                y='feature',
                orientation='h',
                title='Most Frequent Driver Features In Saved Predictions',
                color='count',
                color_continuous_scale=['#fef0d9', '#fdcc8a', '#e34a33'],
            )
            driver_chart.update_layout(
                height=360,
                xaxis_title='Predictions',
                yaxis_title='Feature',
                coloraxis_showscale=False,
            )
            st.plotly_chart(driver_chart, width='stretch', key='manager_driver_frequency_chart')

    st.divider()
    render_section_anchor('manager-launchpad')
    st.markdown('### Decision Launchpad')
    if playbook_summary_df.empty:
        st.info('Add more medium- and high-risk cases to surface the next campaign to launch.')
    else:
        trigger_col1, trigger_col2, trigger_col3 = st.columns(3)
        trigger_col1.metric('Launch Status', campaign_trigger['status'])
        trigger_col2.metric('Top Campaign', campaign_trigger['campaign'])
        trigger_col3.metric(
            'Top Action',
            campaign_trigger['action'] if len(campaign_trigger['action']) <= 28 else campaign_trigger['action'][:28] + '...',
        )
        st.caption(campaign_trigger['reason'])

        launchpad_col1, launchpad_col2 = st.columns([1.1, 0.9])
        with launchpad_col1:
            launchpad_chart = px.bar(
                playbook_summary_df.head(6).sort_values('customers', ascending=True),
                x='customers',
                y='campaign',
                orientation='h',
                color='avg_probability',
                text='revenue_at_risk',
                title='Which Retention Campaign Should Launch Next',
                color_continuous_scale=['#dceefb', '#f4a261', '#d62839'],
            )
            launchpad_chart.update_traces(texttemplate='%{text:,.0f} TND', textposition='outside')
            launchpad_chart.update_layout(
                height=380,
                xaxis_title='Affected customers',
                yaxis_title='Campaign',
                coloraxis_colorbar_title='Avg risk',
            )
            st.plotly_chart(launchpad_chart, width='stretch', key='manager_decision_launchpad_chart')
        with launchpad_col2:
            top_trigger_row = campaign_trigger.get('top_row')
            top_trigger_campaign = None
            if top_trigger_row is not None:
                top_trigger_campaign = top_trigger_row['campaign']
                primary_playbook = {
                    'campaign': top_trigger_row['campaign'],
                    'recommended_action': top_trigger_row['recommended_action'],
                    'primary_reason': get_playbook_template(top_trigger_row['category']).get('label', top_trigger_row['category']),
                    'next_step': get_playbook_template(top_trigger_row['category']).get('next_step', ''),
                    'offer_detail': '',
                }
                render_retention_playbook_block(
                    primary_playbook,
                    title=f"{campaign_trigger['status']} | launch decision",
                )
            for _, playbook_row in playbook_summary_df.head(2).iterrows():
                if top_trigger_campaign is not None and playbook_row['campaign'] == top_trigger_campaign:
                    continue
                playbook = {
                    'campaign': playbook_row['campaign'],
                    'recommended_action': playbook_row['recommended_action'],
                    'primary_reason': get_playbook_template(playbook_row['category']).get('label', playbook_row['category']),
                    'next_step': get_playbook_template(playbook_row['category']).get('next_step', ''),
                    'offer_detail': '',
                }
                render_retention_playbook_block(
                    playbook,
                    title=(
                        f"{int(playbook_row['customers'])} customer(s), "
                        f"{format_currency_tnd(playbook_row['revenue_at_risk'])} at risk"
                    ),
                )

    st.divider()
    st.markdown('### Decision Operations')
    ops_df = full_history_monitoring_df.copy()
    ops_cols = st.columns(4)
    pending_actions = int((ops_df['Action Status'] == 'Pending Action').sum())
    actioned_cases = int((ops_df['Action Status'] == 'Action Taken').sum())
    known_outcomes = int((ops_df['Outcome Status'] == 'Outcome Known').sum())
    ops_cols[0].metric('Pending Action', pending_actions)
    ops_cols[1].metric('Action Taken', actioned_cases)
    ops_cols[2].metric('Outcome Known', known_outcomes)
    ops_cols[3].metric(
        'Action Coverage',
        f"{(actioned_cases / max(len(ops_df), 1)):.1%}",
    )

    action_status_df = (
        ops_df.groupby(['predicted_risk', 'Action Status'])
        .size()
        .reset_index(name='count')
    )
    status_chart = px.bar(
        action_status_df,
        x='predicted_risk',
        y='count',
        color='Action Status',
        barmode='group',
        title='Pending vs Completed Actions by Risk Tier',
        category_orders={'predicted_risk': ['HIGH RISK', 'MEDIUM RISK', 'LOW RISK']},
        color_discrete_map={'Pending Action': '#d62839', 'Action Taken': '#2a9d8f'},
    )
    status_chart.update_layout(height=360)
    st.plotly_chart(status_chart, width='stretch', key='manager_action_status_chart')

    st.markdown('### Retention Opportunity Matrix')
    opportunity_df = ops_df.copy()
    opportunity_df = opportunity_df.dropna(subset=['predicted_probability'])
    if not opportunity_df.empty:
        opportunity_df['opportunity_value'] = np.where(
            opportunity_df['predicted_risk'].eq('HIGH RISK'),
            opportunity_df['MonthlyCharges'],
            np.where(opportunity_df['predicted_risk'].eq('MEDIUM RISK'), opportunity_df['MonthlyCharges'] * 0.5, 0.0),
        )
        opportunity_df['priority_bucket'] = np.select(
            [
                opportunity_df['predicted_probability'] >= 0.75,
                opportunity_df['predicted_probability'] >= 0.55,
            ],
            ['Immediate', 'Near-Term'],
            default='Monitor',
        )
        opportunity_chart = px.scatter(
            opportunity_df,
            x='predicted_probability',
            y='opportunity_value',
            color='priority_bucket',
            size='MonthlyCharges',
            hover_data=[
                column for column in ['id', 'Gouvernorat', 'Contract', 'PaymentMethod', 'Next Best Action']
                if column in opportunity_df.columns
            ],
            title='Risk vs Revenue Opportunity',
            color_discrete_map={'Immediate': '#d62839', 'Near-Term': '#f4a261', 'Monitor': '#2a9d8f'},
        )
        opportunity_chart.update_layout(
            height=380,
            xaxis_title='Predicted churn probability',
            yaxis_title='Revenue opportunity',
            xaxis_tickformat='.0%',
        )
        st.plotly_chart(opportunity_chart, width='stretch', key='manager_retention_opportunity_matrix')
        st.caption('Immediate customers combine high churn probability with the most revenue worth protecting.')


def render_action_center_tab(predictions: pd.DataFrame, model, dv, scaler):
    """Action Center Tab - Manager's actionable customer list"""
    st.subheader('📞 Action Center')
    st.caption('Turn saved predictions into concrete retention actions and track follow-through.')

    monitoring_df = build_decision_support_frame(predictions)

    if monitoring_df.empty:
        st.info('No saved customer predictions are available yet. Run predictions first.')
        return

    saved_df = monitoring_df.copy()
    saved_df["Manager's Action"] = saved_df['Manager Action']

    st.markdown('### Decision Queue')
    queue_cols = st.columns(4)
    queue_cols[0].metric('Pending Action', int((saved_df['Action Status'] == 'Pending Action').sum()))
    queue_cols[1].metric('Action Taken', int((saved_df['Action Status'] == 'Action Taken').sum()))
    queue_cols[2].metric('Outcome Known', int((saved_df['Outcome Status'] == 'Outcome Known').sum()))
    queue_cols[3].metric('Revenue at Risk', format_currency_tnd(saved_df['Revenue at Risk'].sum()))

    pending_df = saved_df[saved_df['Action Status'] == 'Pending Action'].copy()
    if not pending_df.empty:
        st.warning(f"{len(pending_df)} customer(s) are still waiting for a manager decision.")
    else:
        st.success('All current customer predictions already have a manager action assigned.')

    st.markdown('### Assign Manager Action')
    selected_prediction_id = st.selectbox(
        'Choose a saved prediction',
        options=saved_df['id'].tolist(),
        format_func=lambda record_id: (
            f"#{record_id} | "
            f"{saved_df.loc[saved_df['id'] == record_id, 'predicted_risk'].iloc[0]} | "
            f"{saved_df.loc[saved_df['id'] == record_id, 'predicted_probability'].iloc[0]:.1%}"
        ),
        key='action_center_prediction_selector',
    )
    selected_row = saved_df.loc[saved_df['id'] == selected_prediction_id].iloc[0]
    action_guidance = get_manager_action_guidance(selected_row)
    manager_options = get_manager_action_options(selected_row['predicted_risk'])
    recommended_action = action_guidance['recommended_action']
    if recommended_action not in manager_options:
        manager_options = [recommended_action] + manager_options
    saved_action = selected_row.get('manager_action')
    action_brief = build_customer_action_brief(selected_row)
    playbook = action_brief['playbook']
    top_driver_labels = action_brief['top_driver_labels']
    ranked_recommendations = build_ranked_manager_recommendations(selected_row, limit=3)
    default_action = get_default_manager_action(
        saved_action=saved_action,
        manager_options=manager_options,
        ranked_recommendations=ranked_recommendations,
        recommended_action=recommended_action,
    )
    top_ranked_action = ranked_recommendations[0]['action'] if ranked_recommendations else default_action

    choice_col, context_col = st.columns([0.9, 1.1])
    with choice_col:
        st.markdown('**Action Decision**')
        render_decision_badges(action_guidance)
        st.markdown(
            (
                f'<div class="decision-note">Owner: {action_guidance["execution_owner"]}. '
                f'Risk view: {action_guidance["rationale"]}</div>'
            ),
            unsafe_allow_html=True,
        )
        render_customer_action_brief(action_brief)
        st.markdown('**Recommended choices**')
        render_ranked_recommendations(ranked_recommendations, highlighted_action=top_ranked_action)
        with st.form(key=f"manager_action_form_{selected_prediction_id}", clear_on_submit=False):
            if saved_action not in manager_options and default_action == top_ranked_action:
                st.markdown(
                    (
                        '<div class="selection-default-note">'
                        '<span class="selection-default-badge">Recommended</span>'
                        f'<span>The selector starts on {top_ranked_action} from the #1 card.</span>'
                        '</div>'
                    ),
                    unsafe_allow_html=True,
                )
            chosen_action = st.selectbox(
                'Select the manager action',
                options=manager_options,
                index=manager_options.index(default_action),
                format_func=lambda action_name: (
                    f'{action_name} (Recommended)' if action_name == top_ranked_action else action_name
                ),
                key=f"manager_action_choice_{selected_prediction_id}",
            )
            save_action_submitted = st.form_submit_button('Save Manager Action', type='primary')
        if save_action_submitted:
            connection = get_postgres_connection()
            try:
                bootstrap_platform_tables_if_enabled(connection)
                save_manager_action(connection, int(selected_prediction_id), chosen_action)
                clear_runtime_caches()
                st.success(f"Manager action saved for prediction #{selected_prediction_id}.")
                st.rerun()
            except Exception as error:
                st.error(f'Unable to save manager action: {error}')
            finally:
                connection.close()
    with context_col:
        st.markdown('**Customer Context**')
        st.markdown(
            (
                '<div class="context-summary-row">'
                '<div class="context-summary-card">'
                '<div class="context-summary-label">Risk Tier</div>'
                f'<div class="context-summary-value">{get_risk_tier_icon(selected_row["predicted_risk"])} {selected_row["predicted_risk"]}</div>'
                '</div>'
                '<div class="context-summary-card">'
                '<div class="context-summary-label">Churn Probability</div>'
                f'<div class="context-summary-value">{action_guidance["probability_band"]}</div>'
                '</div>'
                '</div>'
            ),
            unsafe_allow_html=True,
        )
        st.markdown('**Main takeaway**')
        st.write(action_brief['summary_text'])
        if playbook.get('offer_detail'):
            st.write(f"Suggested offer: `{playbook['offer_detail']}`")
        if top_driver_labels:
            st.markdown(f'**{action_brief["driver_heading"]}**')
            render_top_driver_chips(top_driver_labels)
        st.write(f"Final chosen action: `{selected_row.get('manager_action') or 'Not selected yet'}`")
        st.write(f"Outcome status: `{selected_row['Outcome Status']}`")

    st.markdown('### Retention Campaign Priority')
    priority_df = saved_df.copy()
    if 'created_at' in priority_df.columns:
        priority_df['created_at'] = pd.to_datetime(priority_df['created_at'], errors='coerce')
    priority_df['priority_score'] = (
        priority_df['predicted_probability'].fillna(0.0) * 100
        + np.where(priority_df['predicted_risk'].eq('HIGH RISK'), 30, np.where(priority_df['predicted_risk'].eq('MEDIUM RISK'), 15, 0))
        + np.where(priority_df['Action Status'].eq('Pending Action'), 20, 0)
        + np.where(priority_df.get('Manager Action', pd.Series(index=priority_df.index)).fillna('Not selected').eq('Not selected'), 10, 0)
    )
    priority_columns = [
        column for column in [
            'id', 'predicted_risk', 'predicted_probability', 'Revenue at Risk',
            'priority_score', 'Recommended Campaign', 'Primary Churn Reason',
            "Manager's Action", 'Gouvernorat', 'Contract', 'PaymentMethod'
        ]
        if column in priority_df.columns
    ]
    priority_campaign_df = (
        priority_df[priority_df['predicted_risk'].isin(['HIGH RISK', 'MEDIUM RISK'])]
        .sort_values(['priority_score', 'predicted_probability'], ascending=[False, False])
        .head(12)
        .copy()
    )
    if priority_campaign_df.empty:
        st.info('Run or seed more production predictions to build a retention campaign queue.')
    else:
        priority_display = priority_campaign_df[priority_columns].copy()
        if 'predicted_probability' in priority_display.columns:
            priority_display['predicted_probability'] = priority_display['predicted_probability'].map(lambda value: f'{value:.1%}')
        if 'Revenue at Risk' in priority_display.columns:
            priority_display['Revenue at Risk'] = priority_display['Revenue at Risk'].map(format_currency_tnd)
        if 'priority_score' in priority_display.columns:
            priority_display['priority_score'] = priority_display['priority_score'].map(lambda value: f'{value:.0f}')
        priority_display = priority_display.rename(columns={'priority_score': 'Retention Priority'})

        st.dataframe(priority_display, width='stretch', hide_index=True)
        st.caption('This table highlights the customers who combine stronger churn urgency, unresolved follow-up, and more revenue worth protecting.')

    # Export button for high-risk customers
    high_risk_list = saved_df[saved_df['predicted_risk'] == 'HIGH RISK'].copy()
    if not high_risk_list.empty:
        export_columns = [
            column for column in [
                'id', 'created_at', 'predicted_probability', 'predicted_risk',
                'Recommended Campaign', 'Primary Churn Reason', "Manager's Action", 'MonthlyCharges', 'Contract',
                'InternetService', 'PaymentMethod', 'Gouvernorat', 'tenure'
            ]
            if column in high_risk_list.columns
        ]
        st.download_button(
            '📥 Export Priority Call List for Marketing',
            data=high_risk_list[export_columns].to_csv(index=False),
            file_name='priority_call_list.csv',
            mime='text/csv',
            type='primary',
        )
    else:
        st.info('No high-risk customers are currently available for the priority call list.')

    # Display columns
    display_columns = [
        column for column in [
            'id', 'created_at', 'predicted_probability', 'predicted_label', 'predicted_risk',
            'Recommended Campaign', 'Primary Churn Reason', "Manager's Action", 'MonthlyCharges', 'Contract',
            'InternetService', 'PaymentMethod', 'Gouvernorat', 'tenure'
        ]
        if column in saved_df.columns
    ]

    # Style function for Manager's Action
    def style_managers_action(value: str) -> str:
        tone = classify_action_tone(value)
        if tone == 'critical':
            return 'background-color: #fde2e4; color: #9d0208; font-weight: 700;'
        if tone == 'warning':
            return 'background-color: #fff3cd; color: #8d6e00; font-weight: 700;'
        if tone == 'informational':
            return 'background-color: #e8f4f8; color: #0f62a8; font-weight: 600;'
        if tone == 'healthy':
            return 'background-color: #d8f3dc; color: #1b4332; font-weight: 700;'
        if tone == 'neutral':
            return 'background-color: #f1f3f5; color: #495057; font-weight: 600;'
        return ''

    styled_df = (
        saved_df[display_columns]
        .sort_values('created_at', ascending=False)
        .style
        .map(style_managers_action, subset=["Manager's Action"])
    )
    st.dataframe(styled_df, width='stretch')

    st.markdown('### Action Summary')
    actioned_df = saved_df[saved_df['manager_action'].notna()].copy()
    if actioned_df.empty:
        st.info('Action summary will appear after manager actions are recorded.')
    else:
        action_mix_df = (
            actioned_df.groupby('manager_action')
            .agg(
                customers=('id', 'count'),
                avg_probability=('predicted_probability', 'mean'),
                revenue_at_risk=('Revenue at Risk', 'sum'),
            )
            .reset_index()
            .sort_values(['customers', 'revenue_at_risk'], ascending=[False, False])
            .head(6)
        )
        lead_action = action_mix_df.iloc[0]
        insight_cols = st.columns(3)
        insight_cols[0].metric('Most Used Action', str(lead_action['manager_action']))
        insight_cols[1].metric('Customers Covered', int(lead_action['customers']))
        insight_cols[2].metric('Avg Risk In Top Action', f"{float(lead_action['avg_probability']):.1%}")
        st.caption('This view highlights where manager effort is currently concentrated and how much churn risk that effort is covering.')

        action_mix_chart = px.bar(
            action_mix_df.sort_values('customers', ascending=True),
            x='customers',
            y='manager_action',
            orientation='h',
            color='avg_probability',
            title='Where Manager Effort Is Going',
            color_continuous_scale=['#d8f3dc', '#f4a261', '#d62839'],
            hover_data={
                'customers': True,
                'avg_probability': ':.1%',
                'revenue_at_risk': None,
                'manager_action': False,
            },
        )
        action_mix_chart.update_layout(
            height=320,
            xaxis_title='Customers',
            yaxis_title='Manager action',
            coloraxis_colorbar_title='Avg risk',
        )
        st.plotly_chart(action_mix_chart, width='stretch', key='manager_action_summary_chart')

def render_technical_lab_tab():
    """Technical Lab Tab - Model experimentation and governance"""
    st.subheader('🧪 Governance & Lab')
    st.caption('Compare a candidate model with production and document whether it should move forward.')
    with st.expander('Technical Delivery Readiness'):
        render_delivery_readiness_panel()

    control_col, summary_col = st.columns([0.95, 1.05])
    with control_col:
        c_exponent = st.slider('log10(C)', min_value=-3.0, max_value=2.0, value=2.0, step=0.1)
        selected_c = round(10 ** c_exponent, 6)
        selected_solver = st.selectbox('Solver', ['lbfgs', 'liblinear', 'saga'])
        if selected_solver == 'lbfgs':
            selected_penalty = 'l2'
        else:
            penalty_options = ['l1', 'l2']
            if selected_solver == 'saga':
                penalty_options.append('elasticnet')
            selected_penalty = st.selectbox('Penalty', penalty_options)
        selected_l1_ratio = None
        if selected_solver == 'saga' and selected_penalty == 'elasticnet':
            selected_l1_ratio = st.slider('Elastic-Net Mix (l1_ratio)', min_value=0.05, max_value=0.95, value=0.50, step=0.05)
        class_weight_options = {
            'balanced': 'balanced',
            'None': None,
            'Positive x2': {0: 1, 1: 2},
            'Positive x3': {0: 1, 1: 3},
            'Positive x4': {0: 1, 1: 4},
        }
        class_weight_label = st.selectbox('Class Weight', list(class_weight_options.keys()))
        selected_class_weight = class_weight_options[class_weight_label]
        candidate_params = {
            'C': selected_c,
            'solver': selected_solver,
            'penalty': selected_penalty,
            'class_weight': selected_class_weight,
            'l1_ratio': selected_l1_ratio,
        }
    with summary_col:
        st.info('This lab is for safe offline comparison before any production change.')
        st.caption('Selection rule: prioritize validation F1, then use ROC-AUC and accuracy as tie-breakers.')

    baseline_result = run_lab_experiment_cached({
        'C': PRODUCTION_BASELINE_PARAMS['C'],
        'solver': PRODUCTION_BASELINE_PARAMS['solver'],
        'penalty': PRODUCTION_BASELINE_PARAMS['penalty'],
        'class_weight': PRODUCTION_BASELINE_PARAMS['class_weight'],
    })
    candidate_result = run_lab_experiment_cached(candidate_params)

    baseline_metrics = baseline_result['metrics']['test']
    candidate_metrics = candidate_result['metrics']['test']
    governance = evaluate_governance_candidate(candidate_metrics, PRODUCTION_BASELINE_METRICS)
    baseline_selection_score = calculate_model_selection_score(baseline_result['metrics']['val'])
    candidate_selection_score = calculate_model_selection_score(candidate_result['metrics']['val'])

    st.divider()
    metric_cols = st.columns(4)
    for column, metric_name, label in zip(metric_cols[:3], ['accuracy', 'f1', 'roc_auc'], ['Accuracy', 'F1-Score', 'ROC-AUC']):
        delta_value = candidate_metrics[metric_name] - baseline_metrics[metric_name]
        column.metric(label, f"{candidate_metrics[metric_name]:.3f}", delta=f'{delta_value:+.3f}')
    metric_cols[3].metric('Selection Score', f'{candidate_selection_score:.3f}', delta=f'{candidate_selection_score - baseline_selection_score:+.3f}')

    chart_col, detail_col = st.columns([1.25, 0.75])
    with chart_col:
        st.plotly_chart(build_metric_comparison_chart(candidate_metrics, baseline_metrics), width='stretch')
    with detail_col:
        st.markdown('### Governance Verdict')
        st.write(f"Decision: `{governance['decision']}`")
        st.write(governance['rationale'])
        st.write(f"Production baseline: `C={PRODUCTION_BASELINE_PARAMS['C']:.0f}`, `solver={PRODUCTION_BASELINE_PARAMS['solver']}`, `penalty={PRODUCTION_BASELINE_PARAMS['penalty']}`")
        st.write(f"Candidate threshold: `{candidate_result['params']['decision_threshold']:.2f}`")
        st.write(f"MLflow tracking URI: `{get_mlflow_tracking_uri()}`")
        st.write(f"MLflow experiment: `{get_mlflow_experiment_name()}`")

    st.plotly_chart(build_precision_recall_figure(candidate_result, baseline_result), width='stretch')

    action_col1, action_col2 = st.columns(2)
    with action_col1:
        if st.button('Log Candidate To MLflow', type='primary'):
            try:
                run_id = log_lab_run_to_mlflow(candidate_result)
                st.session_state.latest_candidate_run_id = run_id
                st.success(f'Candidate logged to MLflow with run ID `{run_id}`.')
            except Exception as error:
                st.error(str(error))
    with action_col2:
        if st.button('Record Governance Decision'):
            connection = get_postgres_connection()
            try:
                bootstrap_platform_tables_if_enabled(connection)
                insert_governance_decision(
                    connection, candidate_name=build_lab_run_name(candidate_result['params']),
                    mlflow_run_id=st.session_state.get('latest_candidate_run_id'),
                    candidate_source='technical_lab', decision=governance['decision'],
                    rationale=governance['rationale'], candidate_params=candidate_result['params'],
                    candidate_metrics=candidate_metrics, production_metrics=PRODUCTION_BASELINE_METRICS,
                )
                st.success('Governance decision recorded in PostgreSQL.')
            except Exception as error:
                st.error(f'Unable to record governance decision: {error}')
            finally:
                connection.close()


def render_deep_dive_analytics_tab(predictions: pd.DataFrame):
    """Deep-Dive Analytics Tab - Business intelligence with icons"""
    st.subheader('🔬 Deep-Dive Analytics')
    st.caption('Detailed business analysis across customer, service, billing, geography, and behavior dimensions using saved prediction history only.')

    if px is None:
        st.error('Plotly is required for the analytics interface.')
        return

    analytics_df = build_saved_predictions_analytics_dataset(predictions)
    if analytics_df.empty:
        st.info('Deep-dive analytics will appear after you save enough production predictions.')
        return

    analytics_df = prepare_analytics_dataset(analytics_df)
    analytics_df = analytics_df.fillna({'Gouvernorat': 'Unknown', 'Data Source': 'Saved Predictions'})

    # KPI Row
    kpi_cols = st.columns(4)
    kpi_cols[0].metric('Dataset Rows', f'{len(analytics_df)}')
    kpi_cols[1].metric('Overall Churn Rate', f"{analytics_df['Churn_Binary'].mean():.1%}")
    kpi_cols[2].metric('Avg Monthly Charges', format_currency_tnd(analytics_df['MonthlyCharges'].mean(), decimals=2))
    kpi_cols[3].metric('Avg Tenure', f"{analytics_df['tenure'].mean():.1f} mo")
    st.caption(f"Analytics are currently based on `{len(analytics_df)}` saved prediction record(s).")

    st.markdown('### Decision Intelligence Overview')
    saved_decision_df = build_decision_support_frame(predictions)
    if saved_decision_df.empty:
        st.info('Saved prediction intelligence will appear here after you run production predictions.')
    else:
        portfolio_df = build_playbook_portfolio_frame(saved_decision_df)
        overview_col1, overview_col2 = st.columns([1.05, 0.95])
        with overview_col1:
            if portfolio_df.empty:
                st.info('Not enough active medium- or high-risk cases to build the decision overview yet.')
            else:
                category_mix = portfolio_df.groupby('category', dropna=False)['customers'].sum().reset_index()
                category_mix['label'] = category_mix['category'].map(
                    lambda value: get_playbook_template(value).get('label', str(value).replace('_', ' ').title())
                )
                category_chart = px.pie(
                    category_mix,
                    values='customers',
                    names='label',
                    title='Current Mix Of Churn Driver Categories',
                    hole=0.45,
                )
                category_chart.update_layout(height=360)
                st.plotly_chart(category_chart, width='stretch', key='deep_dive_driver_category_mix')
        with overview_col2:
            if not portfolio_df.empty:
                top_row = portfolio_df.iloc[0]
                render_retention_playbook_block(
                    {
                        'campaign': top_row['campaign'],
                        'recommended_action': top_row['recommended_action'],
                        'primary_reason': get_playbook_template(top_row['category']).get('label', top_row['category']),
                        'next_step': get_playbook_template(top_row['category']).get('next_step', ''),
                        'offer_detail': '',
                    },
                    title='Top portfolio action',
                )
                st.caption(
                    f"{int(top_row['customers'])} customer(s) currently map to this action path "
                    f"with {format_currency_tnd(top_row['revenue_at_risk'])} in revenue exposure."
                )

    # Analytics blocks with icons
    analytics_blocks = st.tabs(
        ['🌍 Demography', '💳 Services', '📡 Bills & Contracts', '⏳ Tenure & Usage', '🗺️ Geography', '🛠️ Correlations']
    )

    with analytics_blocks[0]:
        render_demography_analysis(analytics_df)
    with analytics_blocks[1]:
        render_services_analysis(analytics_df)
    with analytics_blocks[2]:
        render_billing_analysis(analytics_df)
    with analytics_blocks[3]:
        render_tenure_analysis(analytics_df)
    with analytics_blocks[4]:
        render_geography_analysis(analytics_df)
    with analytics_blocks[5]:
        render_correlation_analysis(analytics_df)


def render_ground_truth_form(monitoring_df: pd.DataFrame, full_history_df: pd.DataFrame | None = None):
    """Ground Truth Form - Confirm real outcomes"""
    ground_truth_df = monitoring_df.copy() if full_history_df is None else full_history_df.copy()
    action_taken_mask = ground_truth_df.get('manager_action', pd.Series(index=ground_truth_df.index, dtype='object')).notna()
    unlabeled = ground_truth_df[ground_truth_df['actual_label'].isna() & action_taken_mask].copy()
    labeled = ground_truth_df[ground_truth_df['actual_label'].notna()].copy()
    action_taken_total = int(action_taken_mask.sum())

    status_cols = st.columns(3)
    status_cols[0].metric('Pending Ground Truth', int(len(unlabeled)))
    status_cols[1].metric('Confirmed Outcomes', int(len(labeled)))
    status_cols[2].metric(
        'Ground Truth Coverage',
        f"{(len(labeled) / max(action_taken_total, 1)):.1%}",
    )

    st.markdown('### Confirm Real Outcomes')
    st.caption('Use this block after some time has passed and you know what really happened to a customer.')
    st.info(
        'Only predictions with a saved manager action appear here, so confirmed outcomes stay tied to cases where the business team actually intervened.'
    )

    actioned_df = ground_truth_df[action_taken_mask].copy()
    selection_df = actioned_df.copy()
    if selection_df.empty:
        st.success('There are no actioned predictions available for outcome confirmation yet.')
        if not labeled.empty:
            preview_columns = [
                column
                for column in ['id', 'created_at', 'predicted_label', 'actual_label', 'ground_truth_source', 'feedback_notes']
                if column in labeled.columns
            ]
            with st.expander('Preview confirmed outcomes'):
                st.dataframe(
                    labeled[preview_columns].sort_values('created_at', ascending=False).head(10),
                    width='stretch',
                )
        return

    if unlabeled.empty:
        st.success('All actioned predictions already have a confirmed outcome.')

    selection_df = selection_df.sort_values('created_at', ascending=False).copy()
    selection_df['selection_label'] = selection_df.apply(
        lambda row: f"ID {row['id']} | {row['created_at']} | predicted={row['predicted_label']} | prob={row['predicted_probability']:.1%}",
        axis=1,
    )
    selection_map = dict(zip(selection_df['selection_label'], selection_df['id']))

    with st.form('ground_truth_form', clear_on_submit=False):
        selected_label = st.selectbox('Choose a past prediction', list(selection_map.keys()))
        selected_row = selection_df.loc[selection_df['selection_label'] == selected_label].iloc[0]
        actual_options = ['Churn', 'No churn']
        current_actual_label = selected_row['actual_label'] if pd.notna(selected_row.get('actual_label')) else 'Churn'
        current_actual_label = current_actual_label if current_actual_label in actual_options else 'Churn'
        actual_label = st.selectbox(
            'What really happened?',
            actual_options,
            index=actual_options.index(current_actual_label),
        )
        feedback_notes = st.text_area(
            'Optional note',
            value='' if pd.isna(selected_row.get('feedback_notes')) else str(selected_row.get('feedback_notes')),
            placeholder='Example: customer renewed contract after retention call',
        )
        submitted = st.form_submit_button('Save Real Outcome')

    if submitted:
        connection = get_postgres_connection()
        try:
            bootstrap_platform_tables_if_enabled(connection)
            update_prediction_ground_truth(
                connection, prediction_id=selection_map[selected_label], actual_label=actual_label,
                ground_truth_source='manual_review', feedback_notes=feedback_notes,
            )
            clear_runtime_caches()
            st.success('Ground-truth outcome stored successfully.')
            st.rerun()
        except Exception as error:
            st.error(f'Unable to store ground truth: {error}')
        finally:
            connection.close()


def render_monitoring_dashboard_tab(predictions: pd.DataFrame, alerts_df: pd.DataFrame, governance_df: pd.DataFrame, model, dv, scaler):
    """Monitoring Dashboard Tab - Production evidence and alerts"""
    st.subheader('Monitoring And Retraining')
    st.caption('Review production evidence, decide whether retraining is needed, and keep the MLOps loop easy to explain.')
    render_mlops_story_panel()
    st.divider()

    active_bundle_metadata = get_bundle_metadata()
    active_baseline_metrics = get_baseline_metrics_from_metadata(active_bundle_metadata)
    filtered_predictions = filter_predictions_for_active_model(predictions, active_bundle_metadata)
    monitoring_df = build_decision_support_frame(filtered_predictions)
    full_history_monitoring_df = build_decision_support_frame(predictions)
    if monitoring_df.empty:
        st.info('No prediction logs are available yet. Run a few production inferences first.')
        return

    live_metrics = calculate_live_metrics(monitoring_df)
    computed_alerts = detect_monitoring_alerts(monitoring_df, active_baseline_metrics)
    latest_governance = governance_df.iloc[0].to_dict() if not governance_df.empty else None
    lifecycle_df = build_lifecycle_frame(
        alerts=computed_alerts,
        governance_decision=latest_governance['decision'] if latest_governance else None,
        feedback_coverage=live_metrics.get('coverage', 0.0),
    )
    retraining = build_retraining_recommendation(computed_alerts)
    high_risk_count = int((monitoring_df['predicted_risk'] == 'HIGH RISK').sum())
    avg_monthly_bill = float(monitoring_df['MonthlyCharges'].mean()) if 'MonthlyCharges' in monitoring_df.columns else 0.0
    revenue_at_risk = calculate_revenue_at_risk_total(monitoring_df)
    model_trust_score = calculate_model_trust_score(computed_alerts)
    recommended_action = get_manager_recommended_action(computed_alerts, high_risk_count, monitoring_df=monitoring_df)
    active_variant_name = active_bundle_metadata.get('variant_name', 'Unknown')
    active_run_id = active_bundle_metadata.get('mlflow_run_id') or active_bundle_metadata.get('run_id') or 'Unknown'
    active_model_version = active_bundle_metadata.get('mlflow_model_version', 'Unknown')
    filtered_from_full_history = len(filtered_predictions) != len(predictions)

    # Summary Status
    summary_status = "Healthy"
    summary_color = "#d8f3dc"
    if any(alert.get('severity') == 'high' for alert in computed_alerts):
        summary_status = "Critical: Immediate Attention Required"
        summary_color = "#fde2e4"
    elif any(alert.get('severity') == 'medium' for alert in computed_alerts):
        summary_status = "Warning: Monitor Closely"
        summary_color = "#fff3cd"
    st.markdown(f'<div style="border-radius:14px;padding:0.8rem 1rem;margin-bottom:0.7rem;background:{summary_color};font-weight:600;">Monitoring Summary: {summary_status}</div>', unsafe_allow_html=True)
    if filtered_from_full_history:
        st.caption('Historical charts still show the broader saved prediction history.')
    else:
        st.caption('Monitoring is using all saved prediction logs for now.')

    # Manager Command Center
    render_section_anchor('monitoring-command-center')
    st.markdown('### Manager Command Center')
    manager_cols = st.columns(3)
    manager_cols[0].metric('Revenue at Risk', format_currency_tnd(revenue_at_risk), help="Sum of monthly revenue for all high-risk customers.")
    manager_cols[1].metric('Model Trust Score', f'{model_trust_score}%', help="Score based on active alerts and model health.")
    manager_cols[2].metric('Recommended Action', recommended_action, help="System-suggested next step based on current risk and alerts.")
    st.markdown(f'<div class="manager-action-banner {get_action_banner_class(recommended_action)}">{recommended_action}</div>', unsafe_allow_html=True)

    # Demo Seed Buttons
    demo_col1, demo_col2 = st.columns(2)
    with demo_col1:
        if st.button('Seed Balanced Demo Predictions'):
            connection = get_postgres_connection()
            try:
                bootstrap_platform_tables_if_enabled(connection)
                inserted_count = seed_demo_predictions(connection, model, dv, scaler)
                st.success(f'Inserted {inserted_count} demo prediction records.')
                st.rerun()
            except Exception as error:
                st.error(f'Unable to seed demo predictions: {error}')
            finally:
                connection.close()
    with demo_col2:
        if st.button('Seed Demo Ground Truth Outcomes'):
            connection = get_postgres_connection()
            try:
                bootstrap_platform_tables_if_enabled(connection)
                updated_count = seed_demo_ground_truth(connection, monitoring_df)
                if updated_count == 0:
                    st.info('No unlabeled prediction records are available for demo outcomes.')
                else:
                    st.success(f'Attached demo outcomes to {updated_count} prediction record(s).')
                    st.rerun()
            except Exception as error:
                st.error(f'Unable to seed demo ground truth: {error}')
            finally:
                connection.close()

    st.divider()
    left_col, right_col = st.columns([1.15, 0.85])
    with left_col:
        st.markdown('### Lifecycle View')
        st.plotly_chart(
            build_lifecycle_figure(lifecycle_df),
            width='stretch',
            key='monitoring_lifecycle_view',
        )
    with right_col:
        render_section_anchor('monitoring-retraining-status')
        st.markdown('### Retraining Status')
        if retraining['status'] == 'Retraining Recommended':
            st.error(retraining['status'])
        elif retraining['status'] == 'Monitor Closely':
            st.warning(retraining['status'])
        else:
            st.success(retraining['status'])
        st.write(retraining['reason'])
        retrain_profile = st.selectbox(
            'Training profile',
            ['quick', 'balanced', 'full'],
            index=0,
            help='Quick is the safest demo option. Full explores more candidate variants.',
            key='monitoring_retrain_profile',
        )
        auto_retrain_enabled = st.checkbox(
            'Auto-retrain when monitoring threshold is reached',
            value=True,
            help='When enabled, local retraining starts automatically once the alert threshold is reached.',
            key='monitoring_auto_retrain_enabled',
        )
        auto_retrain_threshold = st.number_input(
            'Auto-retrain threshold',
            min_value=1,
            max_value=10,
            value=2,
            step=1,
            help='Number of high-severity alerts required before retraining starts automatically.',
            key='monitoring_auto_retrain_threshold',
        )
        st.caption(
            f'Current high-severity alerts: {count_high_severity_alerts(computed_alerts)} '
            f'of {int(auto_retrain_threshold)} required.'
        )
        st.info('For the demo, monitoring and retraining run locally from the app.')

    action_col1, action_col2 = st.columns(2)
    if action_col1.button('Run Monitoring Cycle', type='primary'):
        connection = get_postgres_connection()
        monitoring_stored = False
        try:
            bootstrap_platform_tables_if_enabled(connection)
            replace_monitoring_alerts(connection, computed_alerts)
            monitoring_stored = True
        except Exception as error:
            st.error(f'Unable to store monitoring alerts: {error}')
        finally:
            connection.close()
        if monitoring_stored:
            high_alert_count = count_high_severity_alerts(computed_alerts)
            should_auto_retrain = auto_retrain_enabled and high_alert_count >= int(auto_retrain_threshold)
            if should_auto_retrain:
                with st.spinner(f'Monitoring threshold reached. Launching {retrain_profile} retraining pipeline...'):
                    success, output = launch_training_pipeline(
                        reason='streamlit_monitoring_auto_retrain',
                        profile=retrain_profile,
                        execution_mode='local',
                    )
                if success:
                    clear_runtime_caches(clear_model_bundle=True)
                    st.success('Monitoring alerts were stored and local retraining started successfully.')
                    if output:
                        st.info(output)
                    st.rerun()
                st.error('Monitoring alerts were stored, but automatic retraining failed.')
                if output:
                    st.code(output)
            else:
                clear_runtime_caches()
                st.success('Monitoring alerts refreshed and stored in PostgreSQL.')
                if auto_retrain_enabled:
                    st.info(
                        f'Automatic retraining did not start because only {high_alert_count} high-severity '
                        f'alert(s) are active and the threshold is {int(auto_retrain_threshold)}.'
                    )
                st.rerun()
    retrain_label = 'Retrain Model Now' if retraining['status'] == 'Retraining Recommended' else 'Run Retraining Now'
    if action_col2.button(retrain_label):
        with st.spinner(f'Launching {retrain_profile} training pipeline...'):
            success, output = launch_training_pipeline(
                reason='streamlit_manual_retrain',
                profile=retrain_profile,
                execution_mode='local',
            )
        if success:
            st.cache_data.clear()
            st.success(f'Training pipeline launched successfully with the "{retrain_profile}" profile.')
            if output:
                st.info(output)
            st.rerun()
        else:
            st.error('Training pipeline failed.')
            if output:
                st.code(output)

    st.divider()
    st.markdown('### Probability Drift')
    drift_df = build_probability_drift_frame(full_history_monitoring_df)
    if drift_df.empty:
        st.info('Not enough monitoring data to compute drift yet.')
    else:
        st.plotly_chart(
            build_probability_drift_chart(drift_df),
            width='stretch',
            key='monitoring_probability_drift',
        )

    st.divider()
    perf_col, gov_col = st.columns([1.0, 1.0])
    with perf_col:
        st.markdown('### Live Performance From Confirmed Outcomes')
        if live_metrics.get('sample_size', 0) == 0:
            st.info('No confirmed outcomes yet. Add a few real outcomes below to unlock live post-deployment metrics.')
        else:
            if live_metrics['sample_size'] < 20:
                st.warning(f"These live metrics are based on only {live_metrics['sample_size']} confirmed outcome(s).")
            metric_cols = st.columns(4)
            metric_cols[0].metric('Labeled Records', f"{live_metrics['sample_size']}")
            metric_cols[1].metric('Live Accuracy', f"{live_metrics['accuracy']:.3f}")
            metric_cols[2].metric('Live F1', f"{live_metrics['f1']:.3f}", delta=f"{live_metrics['f1'] - active_baseline_metrics['f1']:+.3f}")
            roc_auc = live_metrics.get('roc_auc')
            metric_cols[3].metric('Live ROC-AUC', f"{roc_auc:.3f}" if pd.notna(roc_auc) else 'N/A')
    with gov_col:
        st.markdown('### Latest Governance Decision')
        if latest_governance is None:
            st.info('No governance decision has been recorded yet.')
        else:
            st.write(f"Candidate: `{latest_governance['candidate_name']}`")
            st.write(f"Decision: `{latest_governance['decision']}`")
            st.write(latest_governance['rationale'])

    st.divider()
    render_ground_truth_form(monitoring_df, full_history_monitoring_df)

    st.divider()
    st.markdown('### Full Saved Prediction History')
    history_col1, history_col2 = st.columns([0.7, 0.3])
    with history_col1:
        history_source_filter = st.selectbox(
            'History source filter',
            options=['All records', 'Manual entry only', 'Demo seeded only'],
            index=0,
            key='full_prediction_history_source_filter',
        )
    with history_col2:
        history_sort_order = st.selectbox(
            'History sort order',
            options=['Newest first', 'Oldest first'],
            index=0,
            key='full_prediction_history_sort_order',
        )

    history_df = full_history_monitoring_df.copy()
    if history_source_filter == 'Manual entry only':
        history_df = history_df[~history_df['Is Demo Record']].copy()
    elif history_source_filter == 'Demo seeded only':
        history_df = history_df[history_df['Is Demo Record']].copy()

    history_columns = [
        column for column in [
            'id', 'created_at', 'Prediction Source', 'predicted_probability', 'predicted_label',
            'predicted_risk', 'Recommended Campaign', 'Primary Churn Reason', 'Manager Action', 'actual_label', 'ground_truth_source',
            'model_version', 'MonthlyCharges', 'tenure', 'Contract', 'InternetService',
            'PaymentMethod', 'Gouvernorat'
        ]
        if column in history_df.columns
    ]
    if history_df.empty:
        st.info('No saved predictions match the selected history filter.')
    else:
        history_display = history_df[history_columns].copy()
        if 'predicted_probability' in history_display.columns:
            history_display['predicted_probability'] = history_display['predicted_probability'].map(
                lambda value: f'{value:.1%}' if pd.notna(value) else ''
            )
        ascending = history_sort_order == 'Oldest first'
        history_display = history_display.sort_values('created_at', ascending=ascending)
        history_summary_cols = st.columns(3)
        history_summary_cols[0].metric('Total Records', len(history_display))
        history_summary_cols[1].metric('Manual Records', int((history_df['Prediction Source'] == 'Manual Entry').sum()))
        history_summary_cols[2].metric('Demo Records', int((history_df['Prediction Source'] == 'Demo Seeded').sum()))
        st.caption(
            f"Showing {len(history_display)} saved prediction record(s) from "
            f"{'oldest to newest' if ascending else 'newest to oldest'}."
        )
        st.download_button(
            'Export Full Prediction History CSV',
            data=history_display.to_csv(index=False),
            file_name='full_prediction_history.csv',
            mime='text/csv',
            type='primary',
            key='export_full_prediction_history_csv',
        )
        st.dataframe(history_display, width='stretch', hide_index=True)


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():
    st.set_page_config(
        page_title='TelCo Churn Prediction',
        page_icon=PAGE_ICON,
        layout='wide',
        initial_sidebar_state='expanded',
    )
    inject_custom_style()

    # Load data
    with st.spinner('Loading production model and platform state...'):
        model, dv, scaler = get_cached_model_bundle()
        predictions, alerts_df, governance_df = load_platform_data()

    with st.sidebar:
        render_sidebar_decision_coach(predictions, alerts_df, governance_df)

    # Render header
    render_header()

    # New tab structure: Managerial Decision Support System
    tabs = st.tabs([
        '📊 Manager Insights',
        '📋 Predictions',
        '📞 Action Center',
        '📈 Monitoring & Retraining',
        '🔬 Deep-Dive Analytics',
        '🧪 Governance & Lab',
    ])

    with tabs[0]:
        render_manager_insights_tab(predictions, alerts_df, governance_df)
    with tabs[1]:
        render_single_prediction_tab(model, dv, scaler)
    with tabs[2]:
        render_action_center_tab(predictions, model, dv, scaler)
    with tabs[3]:
        render_monitoring_dashboard_tab(predictions, alerts_df, governance_df, model, dv, scaler)
    with tabs[4]:
        render_deep_dive_analytics_tab(predictions)
    with tabs[5]:
        render_technical_lab_tab()

if __name__ == '__main__':
    main()
