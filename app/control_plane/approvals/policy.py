from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol


@dataclass(frozen=True)
class ApprovalEvaluationRequest:
    """Immutable context used to determine whether a tool call requires approval."""

    tool_name: str
    tool_metadata: Mapping[str, Any]
    arguments: Mapping[str, Any]
    run_id: str
    step_id: str
    call_id: str
    agent_name: str | None = None
    session_id: str | None = None
    user_id: str | None = None
    principal: str | None = None
    tenant_id: str | None = None

    def __post_init__(self) -> None:
        for field_name, value in (
            ("tool_name", self.tool_name),
            ("run_id", self.run_id),
            ("step_id", self.step_id),
            ("call_id", self.call_id),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must not be empty.")

        for field_name, value in (
            ("agent_name", self.agent_name),
            ("session_id", self.session_id),
            ("user_id", self.user_id),
            ("principal", self.principal),
            ("tenant_id", self.tenant_id),
        ):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{field_name} must not be empty when provided.")

        if not isinstance(self.tool_metadata, Mapping):
            raise TypeError("tool_metadata must be a mapping.")

        if not isinstance(self.arguments, Mapping):
            raise TypeError("arguments must be a mapping.")

        object.__setattr__(self, "tool_metadata", dict(self.tool_metadata))
        object.__setattr__(self, "arguments", dict(self.arguments))


@dataclass(frozen=True)
class ApprovalPolicyDecision:
    """Immutable approval disposition produced by an approval policy."""

    approval_required: bool
    policy_name: str
    policy_version: str
    risk_tier: str
    requested_action: str
    policy_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name, value in (
            ("policy_name", self.policy_name),
            ("policy_version", self.policy_version),
            ("risk_tier", self.risk_tier),
            ("requested_action", self.requested_action),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must not be empty.")

        if not isinstance(self.policy_metadata, Mapping):
            raise TypeError("policy_metadata must be a mapping.")

        object.__setattr__(self, "policy_metadata", dict(self.policy_metadata))


class ApprovalPolicy(Protocol):
    def evaluate(
        self,
        request: ApprovalEvaluationRequest,
    ) -> ApprovalPolicyDecision: ...


class SideEffectApprovalPolicy:
    """
    Require human approval for configured risk tiers when a tool has side effects.

    Tool authorization remains a separate concern. This policy only determines
    whether an already-authorized execution requires human approval.
    """

    def __init__(
        self,
        *,
        approval_risk_tiers: set[str] | frozenset[str],
        policy_name: str = "side-effect-requires-approval",
        policy_version: str = "1.0.0",
    ) -> None:
        if not approval_risk_tiers:
            raise ValueError("approval_risk_tiers must not be empty.")

        if any(not isinstance(value, str) or not value.strip() for value in approval_risk_tiers):
            raise ValueError("approval_risk_tiers must contain only non-empty strings.")

        if not policy_name.strip():
            raise ValueError("policy_name must not be empty.")

        if not policy_version.strip():
            raise ValueError("policy_version must not be empty.")

        self._approval_risk_tiers = frozenset(approval_risk_tiers)
        self._policy_name = policy_name
        self._policy_version = policy_version

    def evaluate(
        self,
        request: ApprovalEvaluationRequest,
    ) -> ApprovalPolicyDecision:
        metadata = request.tool_metadata

        risk_tier = metadata.get("risk_tier", "unknown")
        side_effect = metadata.get("side_effect", False)

        if not isinstance(risk_tier, str) or not risk_tier.strip():
            risk_tier = "unknown"

        requires_approval = side_effect is True and risk_tier in self._approval_risk_tiers

        requested_action = (
            f"Execute tool '{request.tool_name}'"
            if not requires_approval
            else f"Execute side-effecting tool '{request.tool_name}'"
        )

        return ApprovalPolicyDecision(
            approval_required=requires_approval,
            policy_name=self._policy_name,
            policy_version=self._policy_version,
            risk_tier=risk_tier,
            requested_action=requested_action,
            policy_metadata={
                "side_effect": side_effect,
                "capability": metadata.get("capability"),
                "permission_scope": metadata.get("permission_scope"),
            },
        )
