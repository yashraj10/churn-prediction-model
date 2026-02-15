# Churn Prediction & Notification Uplift Model

**Predicting Churn Timing and Measuring Notification Impact for Long-Tenured Wellness Users**

## Overview

This project builds a two-model system for a wellness/fitness subscription app:

1. **Churn Prediction Model** — An XGBoost classifier that detects behavioral drift (declining session frequency, shrinking workout durations, growing inactivity gaps) to identify users likely to cancel within 30 days.
2. **Uplift Analysis** — Measures the causal impact of push notifications on churn reduction across user segments using real notification data.

The analysis focuses on **long-tenured users (180+ day tenure)** to address retention bleed rather than early-stage drop-offs.

## Key Findings

| Metric | Value |
|--------|-------|
| Overall churn rate | ~12.8% |
| Model ROC-AUC (5-fold CV) | **~0.92** |
| Top churn predictor | `days_since_last_workout` (recency drift) |
| Key drift signal | Churners show session decline of −0.42 vs +0.51 for retained |
| Safest cohort | Medium Activity users |
| Highest risk cohort | Low Activity users |
| Notification uplift | **−2.5 pp** churn reduction for notified users |

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

## Project Structure

```
├── Churn_Prediction_Model.ipynb          # Main analysis notebook
├── data/
│   └── agile_churn_raw_v11.xlsx          # Multi-sheet dataset (6 tables)
├── docs/
│   ├── Capstone_Report_Team_3.pdf        # Full written report
│   └── Presentation_Team_3.pdf           # Slide deck
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset

The Excel file contains six sheets:

| Sheet | Description | Key Columns |
|-------|-------------|-------------|
| `users_raw` | User demographics & decision date | `user_id`, `age`, `gender`, `decision_date` |
| `subscriptions_raw` | Billing & tenure info | `billing_cycle`, `first_paid_date`, `paying_status` |
| `sessions_raw` | Workout session logs | `workout_type`, `duration_min`, `calories`, `start_ts` |
| `support_raw` | Support ticket contacts | `channel`, `topic`, `contact_ts` |
| `notifications_raw` | Notification send logs | `sent_ts` |
| `cancellations_raw` | Churn events & reasons | `churn_ts`, `reason` |

## Setup

```bash
git clone https://github.com/<your-username>/churn-prediction-model.git
cd churn-prediction-model
pip install -r requirements.txt
jupyter notebook Churn_Prediction_Model.ipynb
```

## Tech Stack

- **XGBoost** — Gradient boosted tree classifier
- **SHAP** — Model explainability
- **scikit-learn** — Cross-validation, metrics
- **pandas / NumPy** — Data wrangling
- **seaborn / matplotlib** — Visualization

## Team

Capstone Team 3 — MSDS 599

## License

This project is for academic purposes. All data is synthetic.
