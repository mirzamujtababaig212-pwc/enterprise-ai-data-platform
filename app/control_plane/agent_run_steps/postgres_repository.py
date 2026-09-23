from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.control_plane.agent_run_steps.exceptions import (
    DuplicateAgentRunStepError,
)
from app.control_plane.agent_run_steps.models import (
    AgentRunStep,
    AgentRunStepStatus,
)
from app.control_plane.persistence.models import AgentRunStepRecord


class PostgreSQLAgentRunStepsRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def close(self) -> None:
        """Close the repository's database session."""
        self._session.close()

    def create(
        self,
        step: AgentRunStep,
        *,
        commit: bool = True,
    ) -> AgentRunStep:
        try:
            existing = self._session.scalar(
                select(AgentRunStepRecord).where(
                    AgentRunStepRecord.run_id == step.run_id,
                    AgentRunStepRecord.step_id == step.step_id,
                )
            )

            if existing is not None:
                raise DuplicateAgentRunStepError(
                    "agent run step already exists: " f"{step.run_id}/{step.step_id}"
                )

            now = datetime.now(UTC)

            record = AgentRunStepRecord(
                run_id=step.run_id,
                step_id=step.step_id,
                step_index=step.step_index,
                step_type=step.step_type,
                status=step.status.value,
                attempt=step.attempt,
                tool_name=step.tool_name,
                call_id=step.call_id,
                input=step.input,
                output=step.output,
                error=step.error,
                failure_category=step.failure_category,
                started_at=step.started_at,
                completed_at=step.completed_at,
                step_metadata=step.metadata,
                created_at=step.created_at or now,
                updated_at=step.updated_at or now,
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
        run_id: str,
        step_id: str,
    ) -> AgentRunStep | None:
        record = self._session.scalar(
            select(AgentRunStepRecord).where(
                AgentRunStepRecord.run_id == run_id,
                AgentRunStepRecord.step_id == step_id,
            )
        )

        if record is None:
            return None

        return self._to_domain(record)

    def list(
        self,
        run_id: str,
        *,
        status: AgentRunStepStatus | None = None,
        limit: int = 100,
    ) -> list[AgentRunStep]:
        if limit <= 0:
            raise ValueError("limit must be greater than zero.")

        statement = (
            select(AgentRunStepRecord)
            .where(AgentRunStepRecord.run_id == run_id)
            .order_by(
                AgentRunStepRecord.step_index.asc(),
                AgentRunStepRecord.step_id.asc(),
            )
            .limit(limit)
        )

        if status is not None:
            statement = statement.where(
                AgentRunStepRecord.status == status.value,
            )

        records = self._session.scalars(statement).all()

        return [self._to_domain(record) for record in records]

    def transition(
        self,
        run_id: str,
        step_id: str,
        *,
        status: AgentRunStepStatus,
        updated_at: datetime,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
        output=None,
        error: str | None = None,
        failure_category: str | None = None,
        commit: bool = True,
    ) -> AgentRunStep | None:
        try:
            record = self._session.scalar(
                select(AgentRunStepRecord)
                .where(
                    AgentRunStepRecord.run_id == run_id,
                    AgentRunStepRecord.step_id == step_id,
                )
                .with_for_update()
            )

            if record is None:
                return None

            current = self._to_domain(record)
            updated = current.transition_to(status)

            if status == AgentRunStepStatus.RUNNING:
                if started_at is not None:
                    record.started_at = started_at
                record.completed_at = None
            else:
                if completed_at is None:
                    raise ValueError(
                        "completed_at is required for terminal agent run step "
                        f"status: {status.value}"
                    )
                record.completed_at = completed_at

            record.status = updated.status.value
            record.updated_at = updated_at

            if started_at is not None:
                record.started_at = started_at

            if output is not None:
                record.output = output

            record.error = error
            record.failure_category = failure_category

            self._session.flush()

            if commit:
                self._session.commit()

            return self._to_domain(record)

        except Exception:
            if commit:
                self._session.rollback()
            raise

    def bind_execution(
        self,
        run_id: str,
        step_id: str,
        *,
        tool_name: str,
        call_id: str,
        input: object | None = None,
        commit: bool = True,
    ) -> AgentRunStep | None:
        try:
            record = self._session.scalar(
                select(AgentRunStepRecord)
                .where(
                    AgentRunStepRecord.run_id == run_id,
                    AgentRunStepRecord.step_id == step_id,
                )
                .with_for_update()
            )

            if record is None:
                return None

            if record.status != AgentRunStepStatus.RUNNING.value:
                raise ValueError(
                    "agent run step execution binding requires RUNNING status: " f"{record.status}"
                )

            if record.tool_name is not None and record.tool_name != tool_name:
                raise ValueError(
                    "agent run step tool_name is already bound to a different "
                    f"value: {record.tool_name}"
                )

            if record.call_id is not None and record.call_id != call_id:
                raise ValueError(
                    "agent run step call_id is already bound to a different "
                    f"value: {record.call_id}"
                )

            if record.input is not None and record.input != input:
                raise ValueError(
                    "agent run step input is already bound to a different " f"value: {record.input}"
                )

            record.tool_name = tool_name
            record.call_id = call_id
            record.input = input

            self._session.flush()

            if commit:
                self._session.commit()

            return self._to_domain(record)

        except Exception:
            if commit:
                self._session.rollback()
            raise

    def retry(
        self,
        run_id: str,
        step_id: str,
        *,
        updated_at: datetime,
        started_at: datetime | None = None,
        commit: bool = True,
    ) -> AgentRunStep | None:
        try:
            record = self._session.scalar(
                select(AgentRunStepRecord)
                .where(
                    AgentRunStepRecord.run_id == run_id,
                    AgentRunStepRecord.step_id == step_id,
                )
                .with_for_update()
            )

            if record is None:
                return None

            if record.status != AgentRunStepStatus.FAILED.value:
                raise ValueError("agent run step retry requires FAILED status: " f"{record.status}")

            record.status = AgentRunStepStatus.RUNNING.value
            record.attempt += 1
            record.started_at = started_at
            record.completed_at = None
            record.output = None
            record.error = None
            record.failure_category = None
            record.updated_at = updated_at

            self._session.flush()

            if commit:
                self._session.commit()

            return self._to_domain(record)

        except Exception:
            if commit:
                self._session.rollback()
            raise

    @staticmethod
    def _to_domain(record: AgentRunStepRecord) -> AgentRunStep:
        return AgentRunStep(
            run_id=record.run_id,
            step_id=record.step_id,
            step_index=record.step_index,
            step_type=record.step_type,
            status=AgentRunStepStatus(record.status),
            attempt=record.attempt,
            tool_name=record.tool_name,
            call_id=record.call_id,
            input=record.input,
            output=record.output,
            error=record.error,
            failure_category=record.failure_category,
            started_at=record.started_at,
            completed_at=record.completed_at,
            created_at=record.created_at,
            updated_at=record.updated_at,
            metadata=record.step_metadata or {},
        )
