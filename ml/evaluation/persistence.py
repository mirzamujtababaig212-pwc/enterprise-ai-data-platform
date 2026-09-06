from __future__ import annotations

import json

import mlflow

from .policy import QualityGateResult


def persist_quality_gate(result: QualityGateResult) -> None:
    """
    Persist the complete quality-gate decision to the active MLflow run.

    The persisted decision is the governance source of truth used by
    downstream model registration.
    """
    policy_name = result.policy.name

    if not policy_name:
        raise ValueError("Quality gate policy must have a name before persistence")

    mlflow.set_tags(
        {
            "quality_gate_passed": str(result.passed).lower(),
            "quality_gate_policy": policy_name,
            "quality_gate_errors": json.dumps(list(result.errors)),
        }
    )

    thresholds = {
        "quality_gate_min_accuracy": result.policy.min_accuracy,
        "quality_gate_min_precision": result.policy.min_precision,
        "quality_gate_min_recall": result.policy.min_recall,
        "quality_gate_min_f1": result.policy.min_f1,
        "quality_gate_min_roc_auc": result.policy.min_roc_auc,
    }

    mlflow.log_params(
        {key: value if value is not None else "none" for key, value in thresholds.items()}
    )
