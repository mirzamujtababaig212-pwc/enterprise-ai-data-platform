import pytest

from app.control_plane.approvals.policy import (
    ApprovalEvaluationRequest,
    SideEffectApprovalPolicy,
)
from tools.execution.approval import (
    ToolApprovalDecision,
    ToolApprovalDisposition,
)


def make_request(
    *,
    side_effect: bool = True,
    risk_tier: str = "high",
) -> ApprovalEvaluationRequest:
    return ApprovalEvaluationRequest(
        tool_name="send_payment",
        tool_metadata={
            "capability": "payment",
            "risk_tier": risk_tier,
            "side_effect": side_effect,
            "permission_scope": "payments:write",
        },
        arguments={"amount": 100},
        run_id="run-1",
        step_id="step-1",
        call_id="call-1",
        agent_name="payment-agent",
        session_id="session-1",
        user_id="user-1",
        principal="principal-1",
        tenant_id="tenant-1",
    )


def test_side_effect_high_risk_tool_requires_approval() -> None:
    policy = SideEffectApprovalPolicy(approval_risk_tiers={"high", "critical"})

    decision = policy.evaluate(make_request())

    assert decision.approval_required is True
    assert decision.policy_name == "side-effect-requires-approval"
    assert decision.policy_version == "1.0.0"
    assert decision.risk_tier == "high"
    assert decision.requested_action == "Execute side-effecting tool 'send_payment'"
    assert decision.policy_metadata == {
        "side_effect": True,
        "capability": "payment",
        "permission_scope": "payments:write",
    }


def test_non_side_effecting_tool_does_not_require_approval() -> None:
    policy = SideEffectApprovalPolicy(approval_risk_tiers={"high", "critical"})

    decision = policy.evaluate(make_request(side_effect=False))

    assert decision.approval_required is False
    assert decision.risk_tier == "high"


def test_low_risk_side_effecting_tool_does_not_require_approval() -> None:
    policy = SideEffectApprovalPolicy(approval_risk_tiers={"high", "critical"})

    decision = policy.evaluate(make_request(risk_tier="low"))

    assert decision.approval_required is False


def test_missing_metadata_uses_conservative_defaults() -> None:
    policy = SideEffectApprovalPolicy(approval_risk_tiers={"unknown"})

    request = make_request()
    request = ApprovalEvaluationRequest(
        tool_name=request.tool_name,
        tool_metadata={},
        arguments=request.arguments,
        run_id=request.run_id,
        step_id=request.step_id,
        call_id=request.call_id,
    )

    decision = policy.evaluate(request)

    assert decision.approval_required is False
    assert decision.risk_tier == "unknown"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"policy_name": ""},
        {"policy_version": ""},
    ],
)
def test_policy_requires_identity(kwargs: dict[str, str]) -> None:
    with pytest.raises(ValueError):
        SideEffectApprovalPolicy(
            approval_risk_tiers={"high"},
            **kwargs,
        )


def test_policy_requires_at_least_one_risk_tier() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        SideEffectApprovalPolicy(approval_risk_tiers=set())


@pytest.mark.parametrize(
    "disposition",
    list(ToolApprovalDisposition),
)
def test_tool_approval_decision_accepts_all_dispositions(disposition):
    decision = ToolApprovalDecision(
        disposition=disposition,
    )

    assert decision.disposition is disposition
