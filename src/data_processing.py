"""
data_processing.py — Load, clean, and prepare raw data for modeling.
"""

import pandas as pd
import numpy as np


def load_data(path: str) -> dict:
    """Load all six sheets from the Excel dataset."""
    sheets = {
        'users':     pd.read_excel(path, sheet_name='users_raw'),
        'subs':      pd.read_excel(path, sheet_name='subscriptions_raw'),
        'sessions':  pd.read_excel(path, sheet_name='sessions_raw'),
        'support':   pd.read_excel(path, sheet_name='support_raw'),
        'notif':     pd.read_excel(path, sheet_name='notifications_raw'),
        'cancel':    pd.read_excel(path, sheet_name='cancellations_raw'),
    }
    return sheets


def clean_dates(data: dict) -> dict:
    """Convert all timestamp columns to datetime."""
    data['users']['decision_date']  = pd.to_datetime(data['users']['decision_date'])
    data['subs']['first_paid_date'] = pd.to_datetime(data['subs']['first_paid_date'])
    data['sessions']['start_ts']    = pd.to_datetime(data['sessions']['start_ts'])
    data['cancel']['churn_ts']      = pd.to_datetime(data['cancel']['churn_ts'])
    data['notif']['sent_ts']        = pd.to_datetime(data['notif']['sent_ts'])
    data['support']['contact_ts']   = pd.to_datetime(data['support']['contact_ts'])
    return data


def build_base_table(data: dict, churn_window: int = 30) -> pd.DataFrame:
    """
    Merge user, subscription, and cancellation tables.
    Compute tenure and 30-day churn label.
    """
    df = data['users'].merge(data['subs'], on='user_id', how='left')
    df = df.merge(data['cancel'][['user_id', 'churn_ts']], on='user_id', how='left')

    df['tenure_days'] = (df['decision_date'] - df['first_paid_date']).dt.days

    assert df['tenure_days'].min() >= 180, (
        f"Expected 180+ day tenure, got min={df['tenure_days'].min()}"
    )

    churn_mask = (
        (df['churn_ts'].notna()) &
        (df['churn_ts'] >= df['decision_date']) &
        (df['churn_ts'] <= df['decision_date'] + pd.Timedelta(days=churn_window))
    )
    df['target_churn'] = churn_mask.astype(int)
    return df
