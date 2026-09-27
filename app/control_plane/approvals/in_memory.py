from __future__ import annotations

from threading import Lock

from .models import ApprovalOverride, ApprovalRequest, ApprovalStatus
from .repository import ApprovalOverrideRepository, ApprovalRequestRepository


class InMemoryApprovalRequestRepository(ApprovalRequestRepository):
    def __init__(self) -> None:
        self._lock = Lock()
        self._approvals: dict[str, ApprovalRequest] = {}

    def create(
        self,
        approval: ApprovalRequest,
        *,
        commit: bool = True,
    ) -> ApprovalRequest:
        del commit

        with self._lock:
            if approval.approval_id in self._approvals:
                raise ValueError(f"approval request already exists: {approval.approval_id}")

            self._approvals[approval.approval_id] = approval
            return approval

    def get(
        self,
        approval_id: str,
    ) -> ApprovalRequest | None:
        if not approval_id.strip():
            raise ValueError("approval_id must not be empty.")

        with self._lock:
            return self._approvals.get(approval_id)

    def update_status(
        self,
        approval_id: str,
        status: ApprovalStatus,
        *,
        resolved_by: str | None = None,
        resolution_reason: str | None = None,
        commit: bool = True,
    ) -> ApprovalRequest:
        del commit

        with self._lock:
            approval = self._approvals.get(approval_id)

            if approval is None:
                raise KeyError(f"approval request not found: {approval_id}")

            resolved = approval.resolve(
                status,
                resolved_by=resolved_by or "",
                resolution_reason=resolution_reason,
            )

            self._approvals[approval_id] = resolved
            return resolved

    def list_by_run(
        self,
        run_id: str,
    ) -> list[ApprovalRequest]:
        if not run_id.strip():
            raise ValueError("run_id must not be empty.")

        with self._lock:
            return [approval for approval in self._approvals.values() if approval.run_id == run_id]


class InMemoryApprovalOverrideRepository(ApprovalOverrideRepository):
    def __init__(self) -> None:
        self._lock = Lock()
        self._overrides: dict[str, ApprovalOverride] = {}

    def create(
        self,
        override: ApprovalOverride,
        *,
        commit: bool = True,
    ) -> ApprovalOverride:
        del commit

        with self._lock:
            if override.override_id in self._overrides:
                raise ValueError(f"approval override already exists: {override.override_id}")

            if any(
                existing.approval_id == override.approval_id
                for existing in self._overrides.values()
            ):
                raise ValueError(
                    f"approval override already exists for approval: " f"{override.approval_id}"
                )

            self._overrides[override.override_id] = override
            return override

    def get(
        self,
        override_id: str,
    ) -> ApprovalOverride | None:
        if not override_id.strip():
            raise ValueError("override_id must not be empty.")

        with self._lock:
            return self._overrides.get(override_id)

    def get_by_approval(
        self,
        approval_id: str,
    ) -> ApprovalOverride | None:
        if not approval_id.strip():
            raise ValueError("approval_id must not be empty.")

        with self._lock:
            for override in self._overrides.values():
                if override.approval_id == approval_id:
                    return override

        return None

    def list_by_run(
        self,
        run_id: str,
    ) -> list[ApprovalOverride]:
        if not run_id.strip():
            raise ValueError("run_id must not be empty.")

        with self._lock:
            return [override for override in self._overrides.values() if override.run_id == run_id]
