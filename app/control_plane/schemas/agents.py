from __future__ import annotations

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
    agent_name: str
    output: Any

    session_id: str | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )
