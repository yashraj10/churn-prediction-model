# Churn Prediction & Notification Uplift Model

**Predicting Churn Timing and Measuring Notification Impact for Long-Tenured Wellness Users**

---

## Overview

This project builds a two-model system for a wellness/fitness subscription app:

1. **Churn Prediction Model** — An XGBoost classifier that detects behavioral drift (declining session frequency, shrinking workout durations, growing inactivity gaps) to identify users likely to cancel within 30 days.
2. **Uplift Analysis** — Estimates the association between push notifications and churn reduction using **Inverse Propensity Weighting (IPW)** to adjust for confounding, with time-window enforcement on notification timing.

The analysis focuses on **long-tenured users (180+ day tenure)**, verified with an explicit assertion in code.

**Python ≥ 3.10 recommended.**

---

## Results

### Model Performance — ROC-AUC: 0.921

![ROC and Precision-Recall Curves](images/roc_pr_curves.png)

### Feature Importance — Drift Features Dominate

The top predictors are all behavioral drift signals, confirming the model relies on real disengagement patterns rather than static demographics.

![Feature Importance](images/feature_importance.png)

### Drift Signal — Churners vs Retained Users

Churners show clear behavioral decay: longer inactivity gaps, declining workout durations, and dropping session frequency compared to their own baseline.

![Drift Comparison](images/drift_comparison.png)

### Notification Uplift — ~2.5pp Adjusted Churn Reduction

Notifications are associated with lower churn (13.9% → 11.4%), adjusted for baseline covariates via IPW. This is an **observational estimate**, not a randomized experiment — unmeasured confounders may still exist.

![Notification Uplift](images/notification_uplift.png)

---

## Key Findings

| Metric | Value |
|--------|-------|
| Overall churn rate | 12.8% |
| Model ROC-AUC (holdout) | **0.921** |
| Top churn predictor | `days_since_last_workout` |
| Churner session decline | −0.42 vs +0.51 for retained |
| Notification association (IPW-adjusted) | **~−2.5 pp** |
| Safest cohort | Medium Activity users |
| Highest risk cohort | Low Activity users |

---

## Methodology — CRISP-DM

### Drift-Based Feature Engineering

Rather than static point-in-time metrics, the model compares a user's **baseline** activity (45–90 days before decision) against their **recent** activity (last 30 days) to detect disengagement trajectories:

| Feature | Signal |
|---------|--------|
| `session_decline_rate` | Frequency decay between windows |
| `duration_drift` | Declining workout length |
| `inactivity_gap_growth` | Widening gaps between sessions |
| `calorie_trend` | Effort decline |
| `days_since_last_workout` | Immediate recency signal |

### Separated Modeling Concerns

The churn prediction model and the uplift analysis are kept **independent** — the notification flag is not used as a model feature. This prevents conflating "who is at risk" with "what intervention works."

### Uplift Methodology

The notification impact analysis uses three safeguards:

1. **Time-window enforcement** — Only notifications sent 7–30 days before the decision date are counted as treatment. This ensures the notification plausibly preceded the churn decision.
2. **Propensity score adjustment (IPW)** — A logistic regression model estimates each user's probability of receiving a notification based on baseline covariates. Inverse propensity weights adjust for systematic differences between treatment and control groups.
3. **Honest framing** — Results are reported as adjusted associations, not causal effects. A randomized A/B test would be needed to confirm causality.

---

## Project Structure

```
├── Churn_Prediction_Model.ipynb    # Main analysis notebook
├── data/
│   └── agile_churn_raw_v11.xlsx    # Multi-sheet dataset (6 tables)
├── docs/
│   ├── Capstone_Report_Team_3.pdf  # Full written report
│   └── Presentation_Team_3.pptx   # Slide deck
├── images/                          # Result visualizations
│   ├── roc_pr_curves.png
│   ├── feature_importance.png
│   ├── drift_comparison.png
│   └── notification_uplift.png
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset

| Sheet | Description | Key Columns |
|-------|-------------|-------------|
| `users_raw` | Demographics & decision date | `user_id`, `age`, `gender`, `decision_date` |
| `subscriptions_raw` | Billing & tenure | `billing_cycle`, `first_paid_date`, `paying_status` |
| `sessions_raw` | Workout logs | `workout_type`, `duration_min`, `calories`, `start_ts` |
| `support_raw` | Support tickets | `channel`, `topic`, `contact_ts` |
| `notifications_raw` | Notification logs | `sent_ts` |
| `cancellations_raw` | Churn events | `churn_ts`, `reason` |

## Setup

```bash
git clone https://github.com/yashraj10/churn-prediction-model.git
cd churn-prediction-model
pip install -r requirements.txt
jupyter notebook Churn_Prediction_Model.ipynb
```

## Tech Stack

- **XGBoost** — Gradient boosted tree classifier
- **SHAP** — Model explainability
- **scikit-learn** — Cross-validation, metrics, propensity scoring
- **pandas / NumPy** — Data wrangling
- **seaborn / matplotlib** — Visualization

## Team

Capstone Team 3 — MSDS 599

## License

This project is for academic purposes. All data is synthetic.
