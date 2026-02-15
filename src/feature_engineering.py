"""
feature_engineering.py — Drift-based feature construction.

Compares baseline activity (45-90 days before decision) against
recent activity (0-30 days before decision) to detect disengagement.
"""

import pandas as pd
import numpy as np


def _window_agg(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Aggregate session metrics within a time window."""
    agg = df.groupby('user_id').agg(
        count=('session_id', 'count'),
        avg_dur=('duration_min', 'mean'),
        total_cal=('calories', 'sum'),
    ).reset_index()

    gaps = (
        df.sort_values(['user_id', 'start_ts'])
        .groupby('user_id')['start_ts']
        .apply(lambda x: x.diff().dt.days.mean() if len(x) > 1 else np.nan)
        .reset_index(name='avg_gap')
    )
    agg = agg.merge(gaps, on='user_id', how='left')
    return agg.rename(columns={
        c: f'{prefix}_{c}' for c in agg.columns if c != 'user_id'
    })


def compute_drift_features(
    sessions: pd.DataFrame,
    users: pd.DataFrame,
    baseline_range: tuple = (45, 90),
    recent_range: tuple = (0, 30),
) -> pd.DataFrame:
    """
    Compute drift metrics between baseline and recent activity windows.

    Returns a DataFrame with one row per user_id and columns:
        session_decline_rate, duration_drift, inactivity_gap_growth, calorie_trend,
        base_count, recent_count
    """
    sess = sessions.merge(users[['user_id', 'decision_date']], on='user_id', how='left')
    sess['days_before'] = (sess['decision_date'] - sess['start_ts']).dt.days
    sess_pre = sess[sess['days_before'] >= 0]

    baseline = sess_pre[
        (sess_pre['days_before'] >= baseline_range[0]) &
        (sess_pre['days_before'] <= baseline_range[1])
    ]
    recent = sess_pre[sess_pre['days_before'] <= recent_range[1]]

    base_f = _window_agg(baseline, 'base')
    recent_f = _window_agg(recent, 'recent')

    drift = (
        users[['user_id']]
        .merge(base_f, on='user_id', how='left')
        .merge(recent_f, on='user_id', how='left')
    )

    drift = drift.fillna({
        'base_count': 0, 'recent_count': 0,
        'base_avg_dur': 0, 'recent_avg_dur': 0,
        'base_total_cal': 0, 'recent_total_cal': 0,
        'base_avg_gap': 999, 'recent_avg_gap': 999,
    })

    drift['session_decline_rate'] = np.where(
        drift['base_count'] > 0,
        (drift['recent_count'] - drift['base_count']) / drift['base_count'],
        np.where(drift['recent_count'] > 0, 0, -1),
    )
    drift['duration_drift'] = drift['recent_avg_dur'] - drift['base_avg_dur']
    drift['inactivity_gap_growth'] = drift['recent_avg_gap'] - drift['base_avg_gap']
    drift['calorie_trend'] = np.where(
        drift['base_total_cal'] > 0,
        (drift['recent_total_cal'] - drift['base_total_cal']) / drift['base_total_cal'],
        0,
    )
    return drift


def compute_overall_features(
    sessions: pd.DataFrame,
    users: pd.DataFrame,
    support: pd.DataFrame,
    notif: pd.DataFrame,
    notif_window: tuple = (7, 30),
) -> pd.DataFrame:
    """
    Compute total session stats, support tickets, and time-windowed notification flag.
    """
    sess = sessions.merge(users[['user_id', 'decision_date']], on='user_id', how='left')
    sess['days_before'] = (sess['decision_date'] - sess['start_ts']).dt.days
    sess_pre = sess[sess['days_before'] >= 0]

    total = sess_pre.groupby('user_id').agg(
        total_sessions=('session_id', 'count'),
        avg_duration=('duration_min', 'mean'),
        total_calories=('calories', 'sum'),
        days_since_last_workout=('days_before', 'min'),
        workout_type_mode=('workout_type',
                           lambda x: x.mode()[0] if not x.mode().empty else 'Other'),
    ).reset_index()

    support_counts = support.groupby('user_id').size().reset_index(name='support_tickets')
    total = total.merge(support_counts, on='user_id', how='left')

    # Time-window-enforced notification flag
    notif_timed = notif.merge(users[['user_id', 'decision_date']], on='user_id', how='left')
    notif_timed['days_before'] = (notif_timed['decision_date'] - notif_timed['sent_ts']).dt.days
    valid = notif_timed[
        (notif_timed['days_before'] >= notif_window[0]) &
        (notif_timed['days_before'] <= notif_window[1])
    ]
    notified_users = set(valid['user_id'].unique())
    total['received_notification'] = total['user_id'].isin(notified_users).astype(int)

    return total


FEATURE_COLUMNS = [
    'session_decline_rate', 'duration_drift',
    'inactivity_gap_growth', 'calorie_trend',
    'recent_count', 'base_count',
    'total_sessions', 'avg_duration', 'days_since_last_workout',
    'age', 'height_cm', 'weight_kg', 'tenure_days',
    'support_tickets', 'gender_code', 'billing_code',
]


def assemble_model_df(model_df, drift_df, overall_df):
    """Merge all features onto the base table and encode categoricals."""
    df = model_df.merge(drift_df, on='user_id', how='left')
    df = df.merge(overall_df, on='user_id', how='left')

    df['total_sessions'] = df['total_sessions'].fillna(0)
    df['avg_duration'] = df['avg_duration'].fillna(0)
    df['total_calories'] = df['total_calories'].fillna(0)
    df['days_since_last_workout'] = df['days_since_last_workout'].fillna(999)
    df['support_tickets'] = df['support_tickets'].fillna(0)
    df['workout_type_mode'] = df['workout_type_mode'].fillna('None')
    df['received_notification'] = df['received_notification'].fillna(0)

    df['gender_code'] = df['gender'].astype('category').cat.codes
    df['billing_code'] = df['billing_cycle'].astype('category').cat.codes
    df['activity_code'] = df['workout_type_mode'].astype('category').cat.codes

    df['activity_bucket'] = pd.qcut(
        df['total_sessions'], q=3,
        labels=['Low Activity', 'Medium Activity', 'High Activity'],
    )
    return df
