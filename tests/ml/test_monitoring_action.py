from __future__ import annotations

import json

import pytest

from ml.monitoring import (
    DriftAction,
    DriftActionPolicy,
    DriftDecision,
    DriftPolicy,
    DriftStatus,
)


def _decision(overall_status: DriftStatus) -> DriftDecision:
    return DriftDecision(
        policy=DriftPolicy(name="test-policy"),
        feature_status={"feature_a": overall_status},
        overall_status=overall_status,
    )


@pytest.mark.parametrize(
    ("status", "expected_action"),
    [
        (DriftStatus.STABLE, DriftAction.NO_ACTION),
        (DriftStatus.WARNING, DriftAction.INVESTIGATE),
        (DriftStatus.CRITICAL, DriftAction.RETRAIN_REVIEW),
    ],
)
def test_action_policy_maps_drift_status_to_action(
    status: DriftStatus,
    expected_action: DriftAction,
) -> None:
    result = DriftActionPolicy.evaluate(_decision(status))

    assert result.action is expected_action


def test_stable_action_does_not_require_action() -> None:
    result = DriftActionPolicy.evaluate(_decision(DriftStatus.STABLE))

    assert result.requires_action is False


@pytest.mark.parametrize(
    "status",
    [
        DriftStatus.WARNING,
        DriftStatus.CRITICAL,
    ],
)
def test_non_stable_actions_require_action(status: DriftStatus) -> None:
    result = DriftActionPolicy.evaluate(_decision(status))

    assert result.requires_action is True


def test_action_decision_preserves_drift_decision() -> None:
    decision = _decision(DriftStatus.CRITICAL)

    result = DriftActionPolicy.evaluate(decision)

    assert result.decision is decision


def test_action_decision_serializes_to_json_compatible_dict() -> None:
    result = DriftActionPolicy.evaluate(_decision(DriftStatus.CRITICAL))

    payload = result.as_dict()

    assert payload == {
        "schema_version": "v1",
        "action": "retrain_review",
        "requires_action": True,
        "drift_decision": {
            "drift_policy": {
                "name": "test-policy",
                "warning_psi": 0.10,
                "critical_psi": 0.25,
            },
            "feature_status": {
                "feature_a": "critical",
            },
            "overall_status": "critical",
        },
    }

    json.dumps(payload)
