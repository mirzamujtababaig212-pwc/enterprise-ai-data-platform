from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.control_plane.agent_runs.exceptions import (
    AgentRunNotFoundError,
    DuplicateAgentRunError,
)
from app.control_plane.agent_runs.models import AgentRun, AgentRunStatus
from app.control_plane.agent_runs.request_snapshot import AgentRunRequestSnapshot
from app.control_plane.persistence.models import AgentRunRecord


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
            status=run.status.value,
            started_at=run.started_at,
            completed_at=run.completed_at,
            lease_id=run.lease_id,
            lease_expires_at=run.lease_expires_at,
            error_type=run.error_type,
            error_message=run.error_message,
            output=run.output,
            run_metadata=dict(run.metadata),
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
        record.status = run.status.value
        record.started_at = run.started_at
        record.completed_at = run.completed_at
        record.lease_id = run.lease_id
        record.lease_expires_at = run.lease_expires_at
        record.error_type = run.error_type
        record.error_message = run.error_message
        record.output = run.output
        record.run_metadata = dict(run.metadata)
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
    ) -> AgentRun | None:
        statement = (
            update(AgentRunRecord)
            .where(
                AgentRunRecord.run_id == run_id,
                AgentRunRecord.status == AgentRunStatus.FAILED.value,
            )
            .values(
                status=AgentRunStatus.RUNNING.value,
                started_at=started_at,
                completed_at=None,
                error_type=None,
                error_message=None,
                output=None,
                lease_id=lease_id,
                lease_expires_at=lease_expires_at,
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
    ) -> AgentRun | None:
        statement = (
            update(AgentRunRecord)
            .where(
                AgentRunRecord.run_id == run_id,
                AgentRunRecord.status == AgentRunStatus.RUNNING.value,
                AgentRunRecord.lease_expires_at < stale_before,
            )
            .values(
                started_at=started_at,
                completed_at=None,
                error_type=None,
                error_message=None,
                output=None,
                lease_id=lease_id,
                lease_expires_at=lease_expires_at,
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

    def list(
        self,
        *,
        agent_name: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        status: AgentRunStatus | None = None,
        limit: int = 100,
    ) -> list[AgentRun]:
        statement = select(AgentRunRecord)

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
