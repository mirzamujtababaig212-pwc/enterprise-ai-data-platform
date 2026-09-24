from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolGovernanceDecisionRecord:
    """Bounded record of the effective governance decision for a tool."""

    tool_name: str
    decision: str
    tenant_id: str | None = None
    reason: str | None = None
    policy_id: str | None = None
    policy_version: str | None = None
    run_id: str | None = None
    call_id: str | None = None
    agent_name: str | None = None
    session_id: str | None = None
    principal: str | None = None
    details: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.tool_name.strip():
            raise ValueError("tool_name must not be empty.")

        if self.decision not in {"allow", "deny"}:
            raise ValueError("decision must be 'allow' or 'deny'.")

        for field_name, value in (
            ("tenant_id", self.tenant_id),
            ("reason", self.reason),
            ("policy_id", self.policy_id),
            ("policy_version", self.policy_version),
            ("run_id", self.run_id),
            ("call_id", self.call_id),
            ("agent_name", self.agent_name),
            ("session_id", self.session_id),
            ("principal", self.principal),
        ):
            if value is not None and not value.strip():
                raise ValueError(f"{field_name} must not be empty when provided.")

        if self.details is not None:
            object.__setattr__(self, "details", dict(self.details))


class ToolGovernanceDecisionSink(Protocol):
    async def record(
        self,
        record: ToolGovernanceDecisionRecord,
    ) -> None: ...
