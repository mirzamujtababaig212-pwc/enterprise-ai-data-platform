from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from app.control_plane.agent_runs.exceptions import (
    InvalidAgentRunTransitionError,
)
from app.control_plane.agent_runs.request_snapshot import AgentRunRequestSnapshot


class AgentRunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


_ALLOWED_AGENT_RUN_TRANSITIONS: dict[AgentRunStatus, frozenset[AgentRunStatus]] = {
    AgentRunStatus.PENDING: frozenset(
        {
            AgentRunStatus.RUNNING,
            AgentRunStatus.REJECTED,
        }
    ),
    AgentRunStatus.RUNNING: frozenset(
        {
            AgentRunStatus.COMPLETED,
            AgentRunStatus.FAILED,
            AgentRunStatus.CANCELLED,
        }
    ),
    AgentRunStatus.COMPLETED: frozenset(),
    AgentRunStatus.FAILED: frozenset(
        {
            AgentRunStatus.RUNNING,
        }
    ),
    AgentRunStatus.CANCELLED: frozenset(),
    AgentRunStatus.REJECTED: frozenset(),
}


class AgentRunExecutionResult(BaseModel):
    run_id: str = Field(min_length=1)
    response: Any


class AgentRun(BaseModel):
    run_id: str = Field(min_length=1)
    agent_name: str = Field(min_length=1)

    session_id: str | None = None
    user_id: str | None = None
    principal: str | None = None
    tenant_id: str | None = None
    idempotency_key: str | None = None

    status: AgentRunStatus = AgentRunStatus.PENDING

    started_at: datetime | None = None
    completed_at: datetime | None = None

    lease_id: str | None = None
    recovery_attempts: int = 0
    lease_expires_at: datetime | None = None

    cancellation_requested: bool = False
    cancellation_requested_at: datetime | None = None

    error_type: str | None = None
    error_message: str | None = None

    output: Any | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    request_snapshot: AgentRunRequestSnapshot | None = None

    def transition_to(self, status: AgentRunStatus) -> "AgentRun":
        allowed_statuses = _ALLOWED_AGENT_RUN_TRANSITIONS[self.status]

        if status not in allowed_statuses:
            raise InvalidAgentRunTransitionError(
                f"invalid agent run transition: " f"{self.status.value} -> {status.value}"
            )

        return self.model_copy(update={"status": status})
