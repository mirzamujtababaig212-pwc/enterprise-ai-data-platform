from __future__ import annotations

from .policy import EvaluationPolicy


VEHICLE_RISK_POLICY = EvaluationPolicy(
    name="vehicle-risk-v1",
    min_f1=0.90,
    min_roc_auc=0.90,
)


CUSTOMER_CHURN_POLICY = EvaluationPolicy(
    name="customer-churn-v1",
    min_recall=0.80,
    min_f1=0.85,
    min_roc_auc=0.90,
)


LOAN_DEFAULT_POLICY = EvaluationPolicy(
    name="loan-default-v1",
    min_recall=0.90,
    min_f1=0.90,
    min_roc_auc=0.95,
)


MODEL_EVALUATION_POLICIES = {
    "VehicleRiskModel": VEHICLE_RISK_POLICY,
    "CustomerChurnModel": CUSTOMER_CHURN_POLICY,
    "LoanDefaultModel": LOAN_DEFAULT_POLICY,
}


def get_evaluation_policy_for_model(model_name: str) -> EvaluationPolicy:
    try:
        return MODEL_EVALUATION_POLICIES[model_name]
    except KeyError as exc:
        raise ValueError(f"No evaluation policy is configured for model '{model_name}'") from exc


__all__ = [
    "CUSTOMER_CHURN_POLICY",
    "LOAN_DEFAULT_POLICY",
    "MODEL_EVALUATION_POLICIES",
    "VEHICLE_RISK_POLICY",
    "get_evaluation_policy_for_model",
]
