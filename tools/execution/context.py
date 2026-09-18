from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rag.governance import GovernancePolicy


@dataclass(frozen=True)
class ToolExecutionContext:
    """
    Immutable execution context supplied to context-aware tools.

    The context carries execution identity and governed request metadata
    across the centralized tool execution boundary.

    Authorization identity remains a separate `principal` parameter on
    ToolExecutionService.
    """

    run_id: str | None = None
    call_id: str | None = None
    agent_name: str | None = None
    session_id: str | None = None
    user_id: str | None = None
    governance_policy: GovernancePolicy | None = None
    request_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.run_id is not None:
            if not isinstance(self.run_id, str):
                raise TypeError("run_id must be a string or None.")
            if not self.run_id.strip():
                raise ValueError("run_id must not be empty.")

        if self.call_id is not None:
            if not isinstance(self.call_id, str):
                raise TypeError("call_id must be a string or None.")
            if not self.call_id.strip():
                raise ValueError("call_id must not be empty.")

        if self.agent_name is not None:
            if not isinstance(self.agent_name, str):
                raise TypeError("agent_name must be a string or None.")
            if not self.agent_name.strip():
                raise ValueError("agent_name must not be empty.")

        if self.session_id is not None:
            if not isinstance(self.session_id, str):
                raise TypeError("session_id must be a string or None.")
            if not self.session_id.strip():
                raise ValueError("session_id must not be empty.")

        if self.user_id is not None:
            if not isinstance(self.user_id, str):
                raise TypeError("user_id must be a string or None.")
            if not self.user_id.strip():
                raise ValueError("user_id must not be empty.")

        if self.governance_policy is not None and not isinstance(
            self.governance_policy,
            GovernancePolicy,
        ):
            raise TypeError("governance_policy must be a GovernancePolicy or None.")

        object.__setattr__(
            self,
            "request_metadata",
            dict(self.request_metadata),
        )
