from __future__ import annotations

import pytest

from ml.evaluation import (
    EvaluationPolicy,
    EvaluationQualityGate,
    EvaluationResult,
)


def _evaluation() -> EvaluationResult:
    return EvaluationResult(
        accuracy=0.92,
        precision=0.88,
        recall=0.84,
        f1=0.86,
        roc_auc=0.94,
    )


def test_policy_passes_when_all_thresholds_are_met() -> None:
    policy = EvaluationPolicy(
        min_accuracy=0.90,
        min_precision=0.85,
        min_recall=0.80,
        min_f1=0.85,
        min_roc_auc=0.90,
    )

    result = EvaluationQualityGate.evaluate(
        _evaluation(),
        policy,
    )

    assert result.passed is True
    assert result.errors == ()
    assert result.metrics["validation_f1"] == 0.86
    assert result.metrics["validation_roc_auc"] == 0.94


def test_policy_fails_when_threshold_is_not_met() -> None:
    policy = EvaluationPolicy(
        min_f1=0.90,
        min_roc_auc=0.90,
    )

    result = EvaluationQualityGate.evaluate(
        _evaluation(),
        policy,
    )

    assert result.passed is False
    assert len(result.errors) == 1
    assert "validation_f1" in result.errors[0]


def test_policy_reports_multiple_failures() -> None:
    policy = EvaluationPolicy(
        min_accuracy=0.95,
        min_precision=0.95,
        min_f1=0.95,
    )

    result = EvaluationQualityGate.evaluate(
        _evaluation(),
        policy,
    )

    assert result.passed is False
    assert len(result.errors) == 3


def test_missing_roc_auc_fails_when_required() -> None:
    evaluation = EvaluationResult(
        accuracy=0.92,
        precision=0.88,
        recall=0.84,
        f1=0.86,
        roc_auc=None,
    )

    policy = EvaluationPolicy(min_roc_auc=0.80)

    result = EvaluationQualityGate.evaluate(
        evaluation,
        policy,
    )

    assert result.passed is False
    assert "validation_roc_auc is unavailable" in result.errors[0]


@pytest.mark.parametrize(
    "field",
    [
        "min_accuracy",
        "min_precision",
        "min_recall",
        "min_f1",
        "min_roc_auc",
    ],
)
def test_policy_rejects_invalid_threshold(field: str) -> None:
    with pytest.raises(ValueError):
        EvaluationPolicy(**{field: 1.01})

    with pytest.raises(ValueError):
        EvaluationPolicy(**{field: -0.01})
