from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.control_plane.persistence.models import ApprovalRequestRecord

from .models import ApprovalRequest, ApprovalStatus
from .repository import ApprovalRequestRepository


class PostgreSQLApprovalRequestRepository(ApprovalRequestRepository):
    def __init__(self, session: Session) -> None:
        self._session = session

    def close(self) -> None:
        """Close the repository's database session."""
        self._session.close()

    def create(
        self,
        approval: ApprovalRequest,
        *,
        commit: bool = True,
    ) -> ApprovalRequest:
        try:
            existing = self._session.scalar(
                select(ApprovalRequestRecord).where(
                    ApprovalRequestRecord.approval_id == approval.approval_id,
                )
            )

            if existing is not None:
                raise ValueError(f"approval request already exists: {approval.approval_id}")

            now = datetime.now(UTC)

            record = ApprovalRequestRecord(
                approval_id=approval.approval_id,
                run_id=approval.run_id,
                step_id=approval.step_id,
                call_id=approval.call_id,
                tool_name=approval.tool_name,
                idempotency_key=approval.idempotency_key,
                status=approval.status.value,
                policy_name=approval.policy_name,
                policy_version=approval.policy_version,
                risk_tier=approval.risk_tier,
                requested_action=approval.requested_action,
                policy_metadata=dict(approval.policy_metadata),
                resolved_by=approval.resolved_by,
                resolution_reason=approval.resolution_reason,
                created_at=approval.created_at or now,
                updated_at=approval.updated_at or now,
                resolved_at=approval.resolved_at,
            )

            self._session.add(record)
            self._session.flush()

            if commit:
                self._session.commit()

            return self._to_domain(record)

        except Exception:
            if commit:
                self._session.rollback()
            raise

    def get(
        self,
        approval_id: str,
    ) -> ApprovalRequest | None:
        record = self._session.scalar(
            select(ApprovalRequestRecord).where(
                ApprovalRequestRecord.approval_id == approval_id,
            )
        )

        if record is None:
            return None

        return self._to_domain(record)

    def update_status(
        self,
        approval_id: str,
        status: ApprovalStatus,
        *,
        resolved_by: str | None = None,
        resolution_reason: str | None = None,
        commit: bool = True,
    ) -> ApprovalRequest:
        try:
            record = self._session.scalar(
                select(ApprovalRequestRecord)
                .where(
                    ApprovalRequestRecord.approval_id == approval_id,
                )
                .with_for_update()
            )

            if record is None:
                raise KeyError(f"approval request not found: {approval_id}")

            current = self._to_domain(record)

            resolved = current.resolve(
                status,
                resolved_by=resolved_by or "",
                resolution_reason=resolution_reason,
            )

            record.status = resolved.status.value
            record.resolved_by = resolved.resolved_by
            record.resolution_reason = resolved.resolution_reason
            record.resolved_at = resolved.resolved_at
            record.updated_at = resolved.updated_at

            self._session.flush()

            if commit:
                self._session.commit()

            return self._to_domain(record)

        except Exception:
            if commit:
                self._session.rollback()
            raise

    def list_by_run(
        self,
        run_id: str,
    ) -> list[ApprovalRequest]:
        records = self._session.scalars(
            select(ApprovalRequestRecord)
            .where(
                ApprovalRequestRecord.run_id == run_id,
            )
            .order_by(
                ApprovalRequestRecord.created_at.asc(),
                ApprovalRequestRecord.approval_id.asc(),
            )
        ).all()

        return [self._to_domain(record) for record in records]

    @staticmethod
    def _to_domain(
        record: ApprovalRequestRecord,
    ) -> ApprovalRequest:
        return ApprovalRequest(
            approval_id=record.approval_id,
            run_id=record.run_id,
            step_id=record.step_id,
            call_id=record.call_id,
            tool_name=record.tool_name,
            idempotency_key=record.idempotency_key,
            status=ApprovalStatus(record.status),
            policy_name=record.policy_name,
            policy_version=record.policy_version,
            risk_tier=record.risk_tier,
            requested_action=record.requested_action,
            policy_metadata=dict(record.policy_metadata),
            resolved_by=record.resolved_by,
            resolution_reason=record.resolution_reason,
            created_at=record.created_at,
            updated_at=record.updated_at,
            resolved_at=record.resolved_at,
        )
