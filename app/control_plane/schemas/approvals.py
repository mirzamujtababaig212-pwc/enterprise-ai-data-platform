from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.control_plane.approvals.models import ApprovalStatus


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


class ApprovalInboxItem(BaseModel):
    approval_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    step_id: str = Field(min_length=1)
    call_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)
    status: ApprovalStatus
    policy_name: str = Field(min_length=1)
    policy_version: str = Field(min_length=1)
    risk_tier: str = Field(min_length=1)
    requested_action: str = Field(min_length=1)
    policy_metadata: dict[str, Any] = Field(default_factory=dict)
    resolved_by: str | None = None
    resolution_reason: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    resolved_at: datetime | None = None


class ApprovalDecisionResponse(BaseModel):
    run_id: str = Field(min_length=1)
    agent_name: str
    output: Any

    session_id: str | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )
