from __future__ import annotations

from .policy import EvaluationPolicy


VEHICLE_RISK_POLICY = EvaluationPolicy(
    min_f1=0.90,
    min_roc_auc=0.90,
)


CUSTOMER_CHURN_POLICY = EvaluationPolicy(
    min_recall=0.80,
    min_f1=0.85,
    min_roc_auc=0.90,
)


LOAN_DEFAULT_POLICY = EvaluationPolicy(
    min_recall=0.90,
    min_f1=0.90,
    min_roc_auc=0.95,
)


__all__ = [
    "CUSTOMER_CHURN_POLICY",
    "LOAN_DEFAULT_POLICY",
    "VEHICLE_RISK_POLICY",
]
