from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ApprovalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class ApprovalRequest(BaseModel):
    approval_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    step_id: str = Field(min_length=1)
    call_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)

    status: ApprovalStatus = ApprovalStatus.PENDING

    policy_name: str = Field(min_length=1)
    policy_version: str = Field(default="1.0.0", min_length=1)
    risk_tier: str = Field(default="medium", min_length=1)
    requested_action: str = Field(min_length=1)
    policy_metadata: dict[str, Any] = Field(default_factory=dict)

    resolved_by: str | None = None
    resolution_reason: str | None = None

    created_at: datetime | None = None
    updated_at: datetime | None = None
    resolved_at: datetime | None = None

    def resolve(
        self,
        status: ApprovalStatus,
        *,
        resolved_by: str,
        resolution_reason: str | None = None,
        resolved_at: datetime | None = None,
    ) -> "ApprovalRequest":
        if self.status is not ApprovalStatus.PENDING:
            raise ValueError("approval request can only be resolved from pending status")

        if status not in {
            ApprovalStatus.APPROVED,
            ApprovalStatus.REJECTED,
        }:
            raise ValueError("approval request can only be resolved as approved or rejected")

        if not resolved_by.strip():
            raise ValueError("resolved_by must not be empty")

        timestamp = resolved_at or datetime.now(timezone.utc)

        return self.model_copy(
            update={
                "status": status,
                "resolved_by": resolved_by,
                "resolution_reason": resolution_reason,
                "resolved_at": timestamp,
                "updated_at": timestamp,
            }
        )
