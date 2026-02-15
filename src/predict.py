"""
predict.py — Score individual users and explain predictions.
"""

import numpy as np
import pandas as pd


def predict_user(user_id: str, model_df: pd.DataFrame, artifact: dict) -> dict:
    """
    Predict churn probability for a single user and return top risk factors.

    Parameters
    ----------
    user_id : str
        The user to score.
    model_df : pd.DataFrame
        Full feature table (output of assemble_model_df).
    artifact : dict
        Contains 'model' (fitted XGBClassifier) and 'features' (list).

    Returns
    -------
    dict with keys: user_id, churn_probability, risk_level,
                    top_risk_factors, recommendation
    """
    model = artifact['model']
    features = artifact['features']

    row = model_df[model_df['user_id'] == user_id]
    if row.empty:
        return {'error': f'User {user_id} not found'}

    X = row[features].fillna(0)
    prob = float(model.predict_proba(X)[:, 1][0])

    # Risk level
    if prob >= 0.7:
        risk = 'HIGH'
    elif prob >= 0.3:
        risk = 'MEDIUM'
    else:
        risk = 'LOW'

    # Top risk factors via feature contribution
    # Use model feature importances weighted by how extreme each feature value is
    feature_vals = X.iloc[0]
    importances = model.feature_importances_
    contributions = []
    for feat, imp in zip(features, importances):
        val = feature_vals[feat]
        contributions.append({
            'feature': feat,
            'value': round(float(val), 2),
            'importance': round(float(imp), 4),
        })
    contributions.sort(key=lambda x: x['importance'], reverse=True)
    top_factors = contributions[:5]

    # Recommendation
    if risk == 'HIGH':
        rec = 'URGENT: Send personalized re-engagement notification with incentive.'
    elif risk == 'MEDIUM':
        rec = 'MONITOR: Schedule a check-in notification within 7 days.'
    else:
        rec = 'HEALTHY: No intervention needed. Continue standard engagement.'

    return {
        'user_id': user_id,
        'churn_probability': round(prob, 4),
        'risk_level': risk,
        'top_risk_factors': top_factors,
        'recommendation': rec,
    }


def batch_predict(model_df: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    """
    Score all users and return a ranked risk table.
    """
    model = artifact['model']
    features = artifact['features']

    X = model_df[features].fillna(0)
    probs = model.predict_proba(X)[:, 1]

    results = model_df[['user_id']].copy()
    results['churn_probability'] = probs
    results['risk_level'] = pd.cut(
        probs, bins=[0, 0.3, 0.7, 1.0],
        labels=['LOW', 'MEDIUM', 'HIGH'],
    )
    return results.sort_values('churn_probability', ascending=False)
