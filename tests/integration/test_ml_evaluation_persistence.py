from __future__ import annotations

from unittest.mock import patch

import mlflow

from ml.evaluation import (
    EvaluationPolicy,
    QualityGateResult,
    persist_quality_gate,
)


def test_persist_quality_gate_records_passed_decision() -> None:
    policy = EvaluationPolicy(
        name="test-v1",
        min_recall=0.80,
        min_f1=0.85,
    )

    result = QualityGateResult(
        passed=True,
        errors=(),
        metrics={
            "validation_recall": 0.90,
            "validation_f1": 0.88,
        },
        policy=policy,
    )

    with patch.object(mlflow, "set_tags") as set_tags:
        with patch.object(mlflow, "log_params") as log_params:
            persist_quality_gate(result)

    set_tags.assert_called_once_with(
        {
            "quality_gate_passed": "true",
            "quality_gate_policy": "test-v1",
            "quality_gate_errors": "[]",
        }
    )

    log_params.assert_called_once_with(
        {
            "quality_gate_min_accuracy": "none",
            "quality_gate_min_precision": "none",
            "quality_gate_min_recall": 0.80,
            "quality_gate_min_f1": 0.85,
            "quality_gate_min_roc_auc": "none",
        }
    )


def test_persist_quality_gate_records_failed_decision() -> None:
    policy = EvaluationPolicy(
        name="test-v1",
        min_recall=0.90,
    )

    result = QualityGateResult(
        passed=False,
        errors=("validation_recall=0.7000 is below required minimum 0.9000",),
        metrics={
            "validation_recall": 0.70,
        },
        policy=policy,
    )

    with patch.object(mlflow, "set_tags") as set_tags:
        with patch.object(mlflow, "log_params"):
            persist_quality_gate(result)

    tags = set_tags.call_args.args[0]

    assert tags["quality_gate_passed"] == "false"
    assert tags["quality_gate_policy"] == "test-v1"
    assert "validation_recall=0.7000" in tags["quality_gate_errors"]
