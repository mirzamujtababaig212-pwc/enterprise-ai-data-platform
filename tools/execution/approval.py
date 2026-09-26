from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping, Protocol


@dataclass(frozen=True)
class ToolApprovalRequest:
    """Immutable context supplied to the approval boundary."""

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


class ToolApprovalDisposition(StrEnum):
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass(frozen=True)
class ToolApprovalDecision:
    """Decision returned by the approval boundary."""

    disposition: ToolApprovalDisposition
    approval_id: str | None = None
    policy_name: str | None = None
    policy_version: str | None = None
    risk_tier: str | None = None
    requested_action: str | None = None
    policy_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.disposition, ToolApprovalDisposition):
            raise TypeError("disposition must be a ToolApprovalDisposition.")

        for field_name, value in (
            ("approval_id", self.approval_id),
            ("policy_name", self.policy_name),
            ("policy_version", self.policy_version),
            ("risk_tier", self.risk_tier),
            ("requested_action", self.requested_action),
        ):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{field_name} must not be empty when provided.")

        if not isinstance(self.policy_metadata, Mapping):
            raise TypeError("policy_metadata must be a mapping.")

        object.__setattr__(self, "policy_metadata", dict(self.policy_metadata))


class ToolApprovalCoordinator(Protocol):
    async def evaluate(
        self,
        request: ToolApprovalRequest,
    ) -> ToolApprovalDecision: ...
