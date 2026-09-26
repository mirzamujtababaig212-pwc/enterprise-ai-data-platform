from __future__ import annotations

import pytest

from ml.monitoring.policy import (
    DriftEvaluator,
    DriftPolicy,
    DriftStatus,
)


def test_default_policy_classifies_stable_feature() -> None:
    result = DriftEvaluator.evaluate(
        {"feature_a": 0.05},
        DriftPolicy(),
    )

    assert result.overall_status is DriftStatus.STABLE
    assert result.feature_status["feature_a"] is DriftStatus.STABLE
    assert result.stable is True


def test_default_policy_classifies_warning_feature() -> None:
    result = DriftEvaluator.evaluate(
        {"feature_a": 0.15},
        DriftPolicy(),
    )

    assert result.overall_status is DriftStatus.WARNING
    assert result.feature_status["feature_a"] is DriftStatus.WARNING
    assert result.stable is False


def test_default_policy_classifies_critical_feature() -> None:
    result = DriftEvaluator.evaluate(
        {"feature_a": 0.25},
        DriftPolicy(),
    )

    assert result.overall_status is DriftStatus.CRITICAL
    assert result.feature_status["feature_a"] is DriftStatus.CRITICAL


def test_critical_status_takes_precedence() -> None:
    result = DriftEvaluator.evaluate(
        {
            "feature_a": 0.15,
            "feature_b": 0.30,
        },
        DriftPolicy(),
    )

    assert result.feature_status["feature_a"] is DriftStatus.WARNING
    assert result.feature_status["feature_b"] is DriftStatus.CRITICAL
    assert result.overall_status is DriftStatus.CRITICAL


def test_warning_status_takes_precedence_over_stable() -> None:
    result = DriftEvaluator.evaluate(
        {
            "feature_a": 0.05,
            "feature_b": 0.10,
        },
        DriftPolicy(),
    )

    assert result.feature_status["feature_a"] is DriftStatus.STABLE
    assert result.feature_status["feature_b"] is DriftStatus.WARNING
    assert result.overall_status is DriftStatus.WARNING


@pytest.mark.parametrize(
    "warning_psi, critical_psi",
    [
        (-0.01, 0.25),
        (0.10, -0.01),
        (0.30, 0.25),
    ],
)
def test_policy_rejects_invalid_thresholds(
    warning_psi: float,
    critical_psi: float,
) -> None:
    with pytest.raises(ValueError):
        DriftPolicy(
            warning_psi=warning_psi,
            critical_psi=critical_psi,
        )


def test_policy_rejects_blank_name() -> None:
    with pytest.raises(ValueError, match="name"):
        DriftPolicy(name="   ")


def test_named_policy_retains_identity() -> None:
    policy = DriftPolicy(
        name="vehicle-risk-v1",
        warning_psi=0.10,
        critical_psi=0.25,
    )

    assert policy.name == "vehicle-risk-v1"
    assert policy.as_dict() == {
        "name": "vehicle-risk-v1",
        "warning_psi": 0.10,
        "critical_psi": 0.25,
    }


def test_negative_psi_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        DriftEvaluator.evaluate(
            {"feature_a": -0.01},
            DriftPolicy(),
        )


def test_empty_feature_psi_is_stable() -> None:
    result = DriftEvaluator.evaluate(
        {},
        DriftPolicy(),
    )

    assert result.feature_status == {}
    assert result.overall_status is DriftStatus.STABLE


def test_decision_serializes_to_json_compatible_dict() -> None:
    result = DriftEvaluator.evaluate(
        {
            "feature_a": 0.05,
            "feature_b": 0.15,
            "feature_c": 0.30,
        },
        DriftPolicy(name="vehicle-risk-v1"),
    )

    assert result.as_dict() == {
        "drift_policy": {
            "name": "vehicle-risk-v1",
            "warning_psi": 0.10,
            "critical_psi": 0.25,
        },
        "feature_status": {
            "feature_a": "stable",
            "feature_b": "warning",
            "feature_c": "critical",
        },
        "overall_status": "critical",
    }
