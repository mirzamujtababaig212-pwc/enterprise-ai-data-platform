from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ai_platform.agents.models import AgentRequest
from app.control_plane.agent_runs.models import AgentRun


@dataclass(frozen=True)
class AgentDelegationRequest:
    """
    Request to create a durable child-agent delegation.

    Parent hierarchy fields are deliberately supplied by the control-plane
    caller rather than allowing arbitrary hierarchy metadata from the child
    request.
    """

    parent_run_id: str
    child_agent_name: str
    child_request: AgentRequest

    idempotency_key: str | None = None
    causation_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.parent_run_id.strip():
            raise ValueError("parent_run_id must not be empty.")

        if not self.child_agent_name.strip():
            raise ValueError("child_agent_name must not be empty.")

        if self.idempotency_key is not None and not self.idempotency_key.strip():
            raise ValueError("idempotency_key must not be empty when provided.")

        if self.causation_id is not None and not self.causation_id.strip():
            raise ValueError("causation_id must not be empty when provided.")

        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class AgentDelegationResult:
    """
    Durable result of creating or recovering a child-agent delegation.
    """

    child_run: AgentRun
    parent_step_id: str
    created: bool
