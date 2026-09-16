from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class AgentRunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentRun(BaseModel):
    run_id: str = Field(min_length=1)
    agent_name: str = Field(min_length=1)

    session_id: str | None = None
    user_id: str | None = None

    status: AgentRunStatus = AgentRunStatus.PENDING

    started_at: datetime | None = None
    completed_at: datetime | None = None

    error_type: str | None = None
    error_message: str | None = None

    output: Any | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
