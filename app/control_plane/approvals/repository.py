from __future__ import annotations

from typing import Protocol

from .models import ApprovalOverride, ApprovalRequest, ApprovalStatus


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

    def list(
        self,
        *,
        tenant_id: str | None = None,
        principal: str | None = None,
        status: ApprovalStatus | None = None,
        limit: int = 100,
    ) -> list[ApprovalRequest]: ...

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


class ApprovalOverrideRepository(Protocol):
    def create(
        self,
        override: ApprovalOverride,
        *,
        commit: bool = True,
    ) -> ApprovalOverride: ...

    def get(
        self,
        override_id: str,
    ) -> ApprovalOverride | None: ...

    def get_by_approval(
        self,
        approval_id: str,
    ) -> ApprovalOverride | None: ...

    def list_by_run(
        self,
        run_id: str,
    ) -> list[ApprovalOverride]: ...
