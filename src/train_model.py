"""
train_model.py — Train the churn prediction model and save artifacts.
"""

import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score

from src.feature_engineering import FEATURE_COLUMNS


def train(
    model_df: pd.DataFrame,
    features: list = FEATURE_COLUMNS,
    test_size: float = 0.2,
    n_estimators: int = 200,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    random_state: int = 42,
) -> dict:
    """
    Train XGBoost churn classifier with cross-validation and holdout evaluation.

    Returns dict with: model, X_test, y_test, cv_scores, holdout_auc, features
    """
    X = model_df[features].fillna(0)
    y = model_df['target_churn']

    xgb_model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        eval_metric='logloss',
        random_state=random_state,
    )

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(xgb_model, X, y, cv=cv, scoring='roc_auc')

    # Holdout
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )
    xgb_model.fit(X_train, y_train)
    y_probs = xgb_model.predict_proba(X_test)[:, 1]
    holdout_auc = roc_auc_score(y_test, y_probs)

    return {
        'model': xgb_model,
        'features': features,
        'X_test': X_test,
        'y_test': y_test,
        'y_probs': y_probs,
        'cv_scores': cv_scores,
        'holdout_auc': holdout_auc,
    }


def save_model(result: dict, path: str = 'model/churn_model.pkl'):
    """Persist trained model and feature list."""
    artifact = {
        'model': result['model'],
        'features': result['features'],
    }
    with open(path, 'wb') as f:
        pickle.dump(artifact, f)


def load_model(path: str = 'model/churn_model.pkl') -> dict:
    """Load persisted model artifact."""
    with open(path, 'rb') as f:
        return pickle.load(f)
