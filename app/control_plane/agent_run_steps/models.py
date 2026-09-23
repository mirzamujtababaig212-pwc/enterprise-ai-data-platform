from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from app.control_plane.agent_run_steps.exceptions import (
    InvalidAgentRunStepTransitionError,
)


class AgentRunStepStatus(StrEnum):
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    AMBIGUOUS = "ambiguous"
    CANCELLED = "cancelled"


_ALLOWED_AGENT_RUN_STEP_TRANSITIONS: dict[
    AgentRunStepStatus,
    frozenset[AgentRunStepStatus],
] = {
    AgentRunStepStatus.PLANNED: frozenset(
        {
            AgentRunStepStatus.RUNNING,
            AgentRunStepStatus.CANCELLED,
        }
    ),
    AgentRunStepStatus.RUNNING: frozenset(
        {
            AgentRunStepStatus.COMPLETED,
            AgentRunStepStatus.FAILED,
            AgentRunStepStatus.AMBIGUOUS,
            AgentRunStepStatus.CANCELLED,
        }
    ),
    AgentRunStepStatus.COMPLETED: frozenset(),
    AgentRunStepStatus.FAILED: frozenset(
        {
            AgentRunStepStatus.CANCELLED,
        }
    ),
    AgentRunStepStatus.AMBIGUOUS: frozenset(),
    AgentRunStepStatus.CANCELLED: frozenset(),
}


class AgentRunStep(BaseModel):
    run_id: str = Field(min_length=1)
    step_id: str = Field(min_length=1)
    step_index: int = Field(ge=0)
    step_type: str = Field(min_length=1)
    status: AgentRunStepStatus = AgentRunStepStatus.PLANNED
    attempt: int = Field(default=1, ge=1)

    tool_name: str | None = None
    call_id: str | None = None

    input: Any | None = None
    output: Any | None = None

    error: str | None = None
    failure_category: str | None = None

    started_at: datetime | None = None
    completed_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)

    def transition_to(self, status: AgentRunStepStatus) -> "AgentRunStep":
        allowed_statuses = _ALLOWED_AGENT_RUN_STEP_TRANSITIONS[self.status]

        if status not in allowed_statuses:
            raise InvalidAgentRunStepTransitionError(
                "invalid agent run step transition: " f"{self.status.value} -> {status.value}"
            )

        return self.model_copy(update={"status": status})
