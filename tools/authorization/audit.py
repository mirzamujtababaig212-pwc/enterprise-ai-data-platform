from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ToolAuthorizationAuditRecord:
    principal: str
    tool_name: str
    allowed: bool
    reason: str | None = None
    policy_id: str | None = None
    policy_version: str | None = None
    run_id: str | None = None
    call_id: str | None = None
    agent_name: str | None = None
    session_id: str | None = None

    def __post_init__(self) -> None:
        if not self.principal.strip():
            raise ValueError("Principal must not be empty.")
        if not self.tool_name.strip():
            raise ValueError("Tool name must not be empty.")

        for field_name, value in (
            ("reason", self.reason),
            ("policy_id", self.policy_id),
            ("policy_version", self.policy_version),
            ("run_id", self.run_id),
            ("call_id", self.call_id),
            ("agent_name", self.agent_name),
            ("session_id", self.session_id),
        ):
            if value is not None and not value.strip():
                raise ValueError(f"{field_name} must not be empty when provided.")


class ToolAuthorizationAuditSink(Protocol):
    async def record(
        self,
        record: ToolAuthorizationAuditRecord,
    ) -> None: ...
