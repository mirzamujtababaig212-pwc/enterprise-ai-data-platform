from __future__ import annotations

from typing import Protocol

from .models import ApprovalRequest, ApprovalStatus


class ApprovalRequestRepository(Protocol):
    def create(
        self,
        approval: ApprovalRequest,
        *,
        commit: bool = True,
    ) -> ApprovalRequest: ...

    def get(
        self,
        approval_id: str,
    ) -> ApprovalRequest | None: ...

    def update_status(
        self,
        approval_id: str,
        status: ApprovalStatus,
        *,
        resolved_by: str | None = None,
        resolution_reason: str | None = None,
        commit: bool = True,
    ) -> ApprovalRequest: ...

    def list_by_run(
        self,
        run_id: str,
    ) -> list[ApprovalRequest]: ...
