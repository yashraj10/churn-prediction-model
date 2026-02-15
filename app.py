"""
Churn Prediction Dashboard — Streamlit App

Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle

from src.data_processing import load_data, clean_dates, build_base_table
from src.feature_engineering import (
    compute_drift_features, compute_overall_features,
    assemble_model_df, FEATURE_COLUMNS,
)
from src.predict import predict_user, batch_predict

# ── Page config ──
st.set_page_config(
    page_title='Churn Prediction Dashboard',
    page_icon='🔮',
    layout='wide',
)

# ── Styling ──
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px; border-radius: 12px; color: white; text-align: center;
    }
    .risk-high { color: #FF4444; font-weight: bold; font-size: 24px; }
    .risk-medium { color: #FFAA00; font-weight: bold; font-size: 24px; }
    .risk-low { color: #00C851; font-weight: bold; font-size: 24px; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_and_prepare():
    """Load data, engineer features, train model (cached)."""
    data_path = 'data/agile_churn_raw_v11.xlsx'
    if not os.path.exists(data_path):
        st.error(f'Dataset not found at {data_path}')
        st.stop()

    data = clean_dates(load_data(data_path))
    base_df = build_base_table(data)
    drift_df = compute_drift_features(data['sessions'], data['users'])
    overall_df = compute_overall_features(
        data['sessions'], data['users'], data['support'], data['notif']
    )
    model_df = assemble_model_df(base_df, drift_df, overall_df)
    return model_df, data


@st.cache_resource
def get_model(model_df):
    """Train or load the model."""
    model_path = 'model/churn_model.pkl'

    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)

    # Train on the fly
    from src.train_model import train
    result = train(model_df)
    artifact = {'model': result['model'], 'features': result['features']}

    os.makedirs('model', exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(artifact, f)

    return artifact


# ── Load everything ──
model_df, raw_data = load_and_prepare()
artifact = get_model(model_df)

# ══════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════
st.sidebar.title('🔮 Churn Predictor')
page = st.sidebar.radio('Navigate', ['Dashboard', 'User Lookup', 'Risk Table'])

# ══════════════════════════════════════════════════════════
# PAGE 1: Dashboard
# ══════════════════════════════════════════════════════════
if page == 'Dashboard':
    st.title('Churn Prediction Dashboard')
    st.markdown('**Behavioral drift detection for long-tenured wellness users**')

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)

    total = len(model_df)
    churners = model_df['target_churn'].sum()
    churn_rate = model_df['target_churn'].mean()
    notified = model_df['received_notification'].sum()

    col1.metric('Total Users', f'{total:,}')
    col2.metric('Churn Rate', f'{churn_rate:.1%}')
    col3.metric('Churners', f'{churners:,}')
    col4.metric('Notified', f'{notified:,}')

    st.divider()

    # Batch predictions
    risk_df = batch_predict(model_df, artifact)
    merged = model_df[['user_id', 'target_churn', 'activity_bucket']].merge(
        risk_df, on='user_id'
    )

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader('Risk Distribution')
        risk_counts = merged['risk_level'].value_counts()
        colors = {'HIGH': '#FF4444', 'MEDIUM': '#FFAA00', 'LOW': '#00C851'}
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(
            risk_counts.index,
            risk_counts.values,
            color=[colors.get(r, '#888') for r in risk_counts.index],
        )
        for bar, val in zip(bars, risk_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                    f'{val:,}', ha='center', fontweight='bold')
        ax.set_ylabel('Users')
        ax.set_title('Users by Risk Level')
        st.pyplot(fig)

    with col_right:
        st.subheader('Churn Rate by Cohort')
        cohort_rates = model_df.groupby('activity_bucket')['target_churn'].mean()
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        cohort_rates.plot(kind='bar', ax=ax2, color=['#EF5350', '#FFA726', '#66BB6A'])
        ax2.set_ylabel('Churn Rate')
        ax2.set_title('Churn Rate by Activity Cohort')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
        st.pyplot(fig2)

    st.divider()

    # Top at-risk users
    st.subheader('🚨 Top 10 At-Risk Users')
    top_risk = merged.nlargest(10, 'churn_probability')
    st.dataframe(
        top_risk[['user_id', 'churn_probability', 'risk_level', 'activity_bucket']],
        use_container_width=True,
        hide_index=True,
    )


# ══════════════════════════════════════════════════════════
# PAGE 2: User Lookup
# ══════════════════════════════════════════════════════════
elif page == 'User Lookup':
    st.title('🔍 Individual User Lookup')
    st.markdown('Enter a User ID to see their churn risk profile.')

    all_ids = sorted(model_df['user_id'].unique())
    user_id = st.selectbox('Select User ID', all_ids, index=0)

    if st.button('Predict', type='primary') or user_id:
        result = predict_user(user_id, model_df, artifact)

        if 'error' in result:
            st.error(result['error'])
        else:
            # Header with risk color
            risk = result['risk_level']
            prob = result['churn_probability']
            css_class = f'risk-{risk.lower()}'

            st.markdown(f"""
            <div style="text-align: center; padding: 20px;">
                <h2>User: {user_id}</h2>
                <p style="font-size: 18px;">Churn Probability</p>
                <p class="{css_class}">{prob:.1%}</p>
                <p style="font-size: 16px;">Risk Level: <span class="{css_class}">{risk}</span></p>
            </div>
            """, unsafe_allow_html=True)

            # Recommendation
            st.info(f'💡 **Recommendation:** {result["recommendation"]}')

            # Risk factors
            st.subheader('Top Risk Factors')
            factors = result['top_risk_factors']
            factor_df = pd.DataFrame(factors)
            factor_df.columns = ['Feature', 'Value', 'Model Importance']

            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 3.5))
            colors = ['#7B2FBE' if imp > 0.08 else '#CCCCCC'
                       for imp in factor_df['Model Importance']]
            ax.barh(factor_df['Feature'], factor_df['Model Importance'], color=colors)
            ax.set_xlabel('Importance')
            ax.set_title(f'Top Features Driving Prediction for {user_id}')
            ax.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig)

            # User details
            st.subheader('User Profile')
            user_row = model_df[model_df['user_id'] == user_id].iloc[0]
            col1, col2, col3 = st.columns(3)
            col1.metric('Tenure', f'{user_row["tenure_days"]:.0f} days')
            col2.metric('Total Sessions', f'{user_row["total_sessions"]:.0f}')
            col3.metric('Days Since Last Workout',
                        f'{user_row["days_since_last_workout"]:.0f}')

            col4, col5, col6 = st.columns(3)
            col4.metric('Session Decline Rate',
                        f'{user_row.get("session_decline_rate", 0):+.2f}')
            col5.metric('Duration Drift',
                        f'{user_row.get("duration_drift", 0):+.1f} min')
            col6.metric('Support Tickets',
                        f'{user_row["support_tickets"]:.0f}')


# ══════════════════════════════════════════════════════════
# PAGE 3: Risk Table
# ══════════════════════════════════════════════════════════
elif page == 'Risk Table':
    st.title('📊 Full Risk Table')

    risk_df = batch_predict(model_df, artifact)
    merged = model_df[['user_id', 'target_churn', 'activity_bucket',
                        'days_since_last_workout', 'session_decline_rate',
                        'total_sessions', 'received_notification']].merge(
        risk_df, on='user_id'
    )

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        risk_filter = st.multiselect(
            'Filter by Risk Level',
            ['HIGH', 'MEDIUM', 'LOW'],
            default=['HIGH', 'MEDIUM', 'LOW'],
        )
    with col2:
        cohort_filter = st.multiselect(
            'Filter by Cohort',
            ['Low Activity', 'Medium Activity', 'High Activity'],
            default=['Low Activity', 'Medium Activity', 'High Activity'],
        )

    filtered = merged[
        (merged['risk_level'].isin(risk_filter)) &
        (merged['activity_bucket'].isin(cohort_filter))
    ].sort_values('churn_probability', ascending=False)

    st.dataframe(
        filtered.rename(columns={
            'user_id': 'User ID',
            'churn_probability': 'Churn Prob',
            'risk_level': 'Risk',
            'activity_bucket': 'Cohort',
            'days_since_last_workout': 'Days Inactive',
            'session_decline_rate': 'Session Decline',
            'total_sessions': 'Total Sessions',
            'received_notification': 'Notified',
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(f'Showing {len(filtered):,} of {len(merged):,} users')
