from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ApprovalDecisionRequest(BaseModel):
    reason: str | None = Field(
        default=None,
        description="Optional reason recorded with the approval decision.",
    )


class ApprovalOverrideRequest(BaseModel):
    reason: str = Field(
        ...,
        min_length=1,
        description="Reason explaining why the approval was overridden.",
    )


class ApprovalDecisionResponse(BaseModel):
    run_id: str = Field(min_length=1)
    agent_name: str
    output: Any

    session_id: str | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )
