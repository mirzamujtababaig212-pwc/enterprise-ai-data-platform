from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.control_plane.agent_runs.exceptions import (
    AgentRunNotFoundError,
    DuplicateAgentRunError,
)
from app.control_plane.agent_runs.models import AgentRun, AgentRunStatus
from app.control_plane.agent_run_steps.models import AgentRunStepStatus
from app.control_plane.agent_runs.request_snapshot import AgentRunRequestSnapshot
from app.control_plane.persistence.models import AgentRunRecord, AgentRunStepRecord


class PostgreSQLAgentRunRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def create(
        self,
        run: AgentRun,
        *,
        commit: bool = True,
    ) -> AgentRun:
        record = AgentRunRecord(
            run_id=run.run_id,
            agent_name=run.agent_name,
            session_id=run.session_id,
            user_id=run.user_id,
            principal=run.principal,
            tenant_id=run.tenant_id,
            idempotency_key=run.idempotency_key,
            status=run.status.value,
            started_at=run.started_at,
            completed_at=run.completed_at,
            lease_id=run.lease_id,
            lease_expires_at=run.lease_expires_at,
            cancellation_requested=run.cancellation_requested,
            cancellation_requested_at=run.cancellation_requested_at,
            error_type=run.error_type,
            error_message=run.error_message,
            output=run.output,
            run_metadata=dict(run.metadata),
            recovery_attempts=run.recovery_attempts,
            request_snapshot=(
                run.request_snapshot.model_dump(mode="json")
                if run.request_snapshot is not None
                else None
            ),
        )

        try:
            existing = self._session.scalar(
                select(AgentRunRecord).where(AgentRunRecord.run_id == run.run_id)
            )

            if existing is not None:
                raise DuplicateAgentRunError(f"agent run already exists: {run.run_id}")

            self._session.add(record)
            self._session.flush()

            if commit:
                self._session.commit()
        except DuplicateAgentRunError:
            if commit:
                self._session.rollback()
            raise
        except IntegrityError as exc:
            if commit:
                self._session.rollback()
            raise DuplicateAgentRunError(f"agent run already exists: {run.run_id}") from exc
        except Exception:
            if commit:
                self._session.rollback()
            raise

        return run

    def get(self, run_id: str) -> AgentRun | None:
        record = self._session.scalar(select(AgentRunRecord).where(AgentRunRecord.run_id == run_id))

        if record is None:
            return None

        return self._to_domain(record)

    def get_for_tenant(
        self,
        run_id: str,
        tenant_id: str,
    ) -> AgentRun | None:
        record = self._session.scalar(
            select(AgentRunRecord).where(
                AgentRunRecord.run_id == run_id,
                AgentRunRecord.tenant_id == tenant_id,
            )
        )

        if record is None:
            return None

        return self._to_domain(record)

    def get_by_idempotency_key(
        self,
        tenant_id: str,
        user_id: str,
        idempotency_key: str,
    ) -> AgentRun | None:
        record = self._session.scalar(
            select(AgentRunRecord).where(
                AgentRunRecord.tenant_id == tenant_id,
                AgentRunRecord.user_id == user_id,
                AgentRunRecord.idempotency_key == idempotency_key,
            )
        )

        if record is None:
            return None

        return self._to_domain(record)

    def update(
        self,
        run: AgentRun,
        *,
        commit: bool = True,
    ) -> AgentRun:
        record = self._session.scalar(
            select(AgentRunRecord).where(AgentRunRecord.run_id == run.run_id)
        )

        if record is None:
            raise AgentRunNotFoundError(f"agent run not found: {run.run_id}")

        record.agent_name = run.agent_name
        record.session_id = run.session_id
        record.user_id = run.user_id
        record.principal = run.principal
        record.tenant_id = run.tenant_id
        record.idempotency_key = run.idempotency_key
        record.status = run.status.value
        record.started_at = run.started_at
        record.completed_at = run.completed_at
        record.lease_id = run.lease_id
        record.lease_expires_at = run.lease_expires_at
        record.cancellation_requested = run.cancellation_requested
        record.cancellation_requested_at = run.cancellation_requested_at
        record.error_type = run.error_type
        record.error_message = run.error_message
        record.output = run.output
        record.run_metadata = dict(run.metadata)
        record.recovery_attempts = run.recovery_attempts
        record.request_snapshot = (
            run.request_snapshot.model_dump(mode="json")
            if run.request_snapshot is not None
            else None
        )

        try:
            self._session.flush()

            if commit:
                self._session.commit()
        except Exception:
            if commit:
                self._session.rollback()
            raise

        return run

    def claim_for_recovery(
        self,
        run_id: str,
        *,
        started_at: datetime,
        lease_id: str,
        lease_expires_at: datetime,
        max_recovery_attempts: int,
    ) -> AgentRun | None:
        statement = (
            update(AgentRunRecord)
            .where(
                AgentRunRecord.run_id == run_id,
                AgentRunRecord.status == AgentRunStatus.FAILED.value,
                AgentRunRecord.cancellation_requested.is_(False),
                AgentRunRecord.recovery_attempts < max_recovery_attempts,
            )
            .values(
                status=AgentRunStatus.RUNNING.value,
                completed_at=None,
                error_type=None,
                error_message=None,
                output=None,
                lease_id=lease_id,
                lease_expires_at=lease_expires_at,
                recovery_attempts=AgentRunRecord.recovery_attempts + 1,
            )
        )

        try:
            result = self._session.execute(statement)

            if result.rowcount != 1:
                return None

            self._session.commit()
        except Exception:
            self._session.rollback()
            raise

        return self.get(run_id)

    def heartbeat(
        self,
        run_id: str,
        *,
        lease_id: str,
        lease_expires_at: datetime,
    ) -> AgentRun | None:
        statement = (
            update(AgentRunRecord)
            .where(
                AgentRunRecord.run_id == run_id,
                AgentRunRecord.status == AgentRunStatus.RUNNING.value,
                AgentRunRecord.lease_id == lease_id,
            )
            .values(
                lease_expires_at=lease_expires_at,
            )
        )

        try:
            result = self._session.execute(statement)

            if result.rowcount != 1:
                self._session.rollback()
                return None

            self._session.commit()
        except Exception:
            self._session.rollback()
            raise

        return self.get(run_id)

    def claim_expired_running_run(
        self,
        run_id: str,
        *,
        stale_before: datetime,
        started_at: datetime,
        lease_id: str,
        lease_expires_at: datetime,
        max_recovery_attempts: int,
    ) -> AgentRun | None:
        statement = (
            update(AgentRunRecord)
            .where(
                AgentRunRecord.run_id == run_id,
                AgentRunRecord.status == AgentRunStatus.RUNNING.value,
                AgentRunRecord.lease_expires_at < stale_before,
                AgentRunRecord.cancellation_requested.is_(False),
                AgentRunRecord.recovery_attempts < max_recovery_attempts,
            )
            .values(
                completed_at=None,
                error_type=None,
                error_message=None,
                output=None,
                lease_id=lease_id,
                lease_expires_at=lease_expires_at,
                recovery_attempts=AgentRunRecord.recovery_attempts + 1,
            )
        )

        try:
            result = self._session.execute(statement)

            if result.rowcount != 1:
                return None

            self._session.commit()
        except Exception:
            self._session.rollback()
            raise

        return self.get(run_id)

    def fail_recovery_exhausted(
        self,
        run_id: str,
        *,
        completed_at: datetime,
        max_recovery_attempts: int,
        error_type: str,
        error_message: str,
    ) -> AgentRun | None:
        statement = (
            update(AgentRunRecord)
            .where(
                AgentRunRecord.run_id == run_id,
                AgentRunRecord.status == AgentRunStatus.RUNNING.value,
                AgentRunRecord.cancellation_requested.is_(False),
                AgentRunRecord.lease_expires_at.is_not(None),
                AgentRunRecord.recovery_attempts >= max_recovery_attempts,
                AgentRunRecord.lease_expires_at < completed_at,
            )
            .values(
                status=AgentRunStatus.FAILED.value,
                completed_at=completed_at,
                error_type=error_type,
                error_message=error_message,
                lease_id=None,
                lease_expires_at=None,
            )
            .returning(AgentRunRecord)
        )

        try:
            record = self._session.execute(statement).scalar_one_or_none()
            self._session.commit()
        except Exception:
            self._session.rollback()
            raise

        if record is None:
            return None

        return self._to_domain(record)

    def list_expired_running_runs(
        self,
        *,
        stale_before: datetime,
        limit: int = 100,
    ) -> list[AgentRun]:
        if limit <= 0:
            raise ValueError("limit must be greater than zero.")

        statement = (
            select(AgentRunRecord)
            .where(
                AgentRunRecord.status == AgentRunStatus.RUNNING.value,
                AgentRunRecord.lease_expires_at.is_not(None),
                AgentRunRecord.lease_expires_at < stale_before,
                AgentRunRecord.cancellation_requested.is_(False),
            )
            .order_by(
                AgentRunRecord.lease_expires_at.asc(),
                AgentRunRecord.run_id.asc(),
            )
            .limit(limit)
        )

        records = self._session.scalars(statement).all()

        return [self._to_domain(record) for record in records]

    def complete_if_owner(
        self,
        run_id: str,
        *,
        lease_id: str,
        completed_at: datetime,
        output,
    ) -> AgentRun | None:
        statement = (
            update(AgentRunRecord)
            .where(
                AgentRunRecord.run_id == run_id,
                AgentRunRecord.status == AgentRunStatus.RUNNING.value,
                AgentRunRecord.lease_id == lease_id,
                AgentRunRecord.lease_expires_at > completed_at,
                ~select(AgentRunStepRecord.run_id)
                .where(
                    AgentRunStepRecord.run_id == AgentRunRecord.run_id,
                    AgentRunStepRecord.status != AgentRunStepStatus.COMPLETED.value,
                )
                .exists(),
            )
            .values(
                status=AgentRunStatus.COMPLETED.value,
                completed_at=completed_at,
                output=output,
                lease_id=None,
                lease_expires_at=None,
            )
        )

        try:
            result = self._session.execute(statement)

            if result.rowcount != 1:
                self._session.rollback()
                return None

            self._session.commit()
        except Exception:
            self._session.rollback()
            raise

        return self.get(run_id)

    def fail_if_owner(
        self,
        run_id: str,
        *,
        lease_id: str,
        completed_at: datetime,
        error_type: str,
        error_message: str,
    ) -> AgentRun | None:
        statement = (
            update(AgentRunRecord)
            .where(
                AgentRunRecord.run_id == run_id,
                AgentRunRecord.status == AgentRunStatus.RUNNING.value,
                AgentRunRecord.lease_id == lease_id,
                AgentRunRecord.lease_expires_at > completed_at,
            )
            .values(
                status=AgentRunStatus.FAILED.value,
                completed_at=completed_at,
                error_type=error_type,
                error_message=error_message,
                lease_id=None,
                lease_expires_at=None,
            )
        )

        try:
            result = self._session.execute(statement)

            if result.rowcount != 1:
                self._session.rollback()
                return None

            self._session.commit()
        except Exception:
            self._session.rollback()
            raise

        return self.get(run_id)

    def cancel_if_owner(
        self,
        run_id: str,
        *,
        lease_id: str,
        completed_at: datetime,
    ) -> AgentRun | None:
        statement = (
            update(AgentRunRecord)
            .where(
                AgentRunRecord.run_id == run_id,
                AgentRunRecord.status == AgentRunStatus.RUNNING.value,
                AgentRunRecord.lease_id == lease_id,
                AgentRunRecord.lease_expires_at > completed_at,
            )
            .values(
                status=AgentRunStatus.CANCELLED.value,
                completed_at=completed_at,
                lease_id=None,
                lease_expires_at=None,
            )
        )

        try:
            result = self._session.execute(statement)

            if result.rowcount != 1:
                self._session.rollback()
                return None

            self._session.commit()
        except Exception:
            self._session.rollback()
            raise

        return self.get(run_id)

    def request_cancellation(
        self,
        run_id: str,
        *,
        requested_at: datetime,
    ) -> AgentRun | None:
        statement = (
            update(AgentRunRecord)
            .where(
                AgentRunRecord.run_id == run_id,
                AgentRunRecord.status == AgentRunStatus.RUNNING.value,
            )
            .values(
                cancellation_requested=True,
                cancellation_requested_at=func.coalesce(
                    AgentRunRecord.cancellation_requested_at,
                    requested_at,
                ),
            )
        )

        try:
            result = self._session.execute(statement)

            if result.rowcount != 1:
                self._session.rollback()
                return None

            self._session.commit()

        except Exception:
            self._session.rollback()
            raise

        return self.get(run_id)

    def list(
        self,
        *,
        tenant_id: str | None = None,
        agent_name: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        status: AgentRunStatus | None = None,
        limit: int = 100,
    ) -> list[AgentRun]:
        statement = select(AgentRunRecord)

        if tenant_id is not None:
            statement = statement.where(AgentRunRecord.tenant_id == tenant_id)

        if agent_name is not None:
            statement = statement.where(AgentRunRecord.agent_name == agent_name)

        if session_id is not None:
            statement = statement.where(AgentRunRecord.session_id == session_id)

        if user_id is not None:
            statement = statement.where(AgentRunRecord.user_id == user_id)

        if status is not None:
            statement = statement.where(AgentRunRecord.status == status.value)

        statement = statement.order_by(
            AgentRunRecord.started_at.desc(),
            AgentRunRecord.run_id.desc(),
        ).limit(limit)

        records = self._session.scalars(statement).all()

        return [self._to_domain(record) for record in records]

    @staticmethod
    def _to_domain(record: AgentRunRecord) -> AgentRun:
        return AgentRun(
            run_id=record.run_id,
            agent_name=record.agent_name,
            session_id=record.session_id,
            user_id=record.user_id,
            principal=record.principal,
            tenant_id=record.tenant_id,
            idempotency_key=record.idempotency_key,
            status=AgentRunStatus(record.status),
            started_at=(
                _ensure_aware(record.started_at) if record.started_at is not None else None
            ),
            completed_at=(
                _ensure_aware(record.completed_at) if record.completed_at is not None else None
            ),
            lease_id=record.lease_id,
            lease_expires_at=(
                _ensure_aware(record.lease_expires_at)
                if record.lease_expires_at is not None
                else None
            ),
            recovery_attempts=record.recovery_attempts,
            cancellation_requested=record.cancellation_requested,
            cancellation_requested_at=(
                _ensure_aware(record.cancellation_requested_at)
                if record.cancellation_requested_at is not None
                else None
            ),
            error_type=record.error_type,
            error_message=record.error_message,
            output=record.output,
            metadata=dict(record.run_metadata),
            request_snapshot=(
                AgentRunRequestSnapshot.model_validate(record.request_snapshot)
                if record.request_snapshot is not None
                else None
            ),
        )


def _ensure_aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        # SQLite does not preserve timezone information for DateTime(timezone=True).
        # PostgreSQL returns an aware datetime for the production schema.
        return value.replace(tzinfo=UTC)

    return value
