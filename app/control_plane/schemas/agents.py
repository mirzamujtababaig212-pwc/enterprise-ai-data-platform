from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class AgentRunRequest(BaseModel):
    input: str = Field(
        ...,
        min_length=1,
        description="User input supplied to the agent.",
    )

    session_id: str | None = Field(
        default=None,
        description="Optional conversation/session identifier.",
    )

    user_id: str | None = Field(
        default=None,
        description="Optional user identifier propagated to the LLM Gateway.",
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional request metadata.",
    )


class AgentRunResponse(BaseModel):
    run_id: str = Field(min_length=1)
    agent_name: str
    output: Any

    session_id: str | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


class AgentRunCancellationResponse(BaseModel):
    run_id: str = Field(min_length=1)
    status: str
    message: str


class AgentRunDetailResponse(BaseModel):
    run_id: str
    agent_name: str
    status: str
    session_id: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    output: Any | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentRunListResponse(BaseModel):
    runs: list[AgentRunDetailResponse]


class AgentRunEventResponse(BaseModel):
    event_type: str
    agent_name: str
    run_id: str | None = None
    session_id: str | None = None
    user_id: str | None = None

    tool_round: int | None = None
    tool_name: str | None = None
    call_id: str | None = None

    step_id: str | None = None
    step_index: int | None = None
    step_name: str | None = None
    attempt: int | None = None

    provider: str | None = None
    model: str | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentRunEventListResponse(BaseModel):
    events: list[AgentRunEventResponse]
    next_cursor: str | None = None
    has_more: bool = False


class AgentRunStepResponse(BaseModel):
    run_id: str
    step_id: str
    step_index: int
    step_type: str
    status: str
    attempt: int
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


class AgentRunStepListResponse(BaseModel):
    steps: list[AgentRunStepResponse]
