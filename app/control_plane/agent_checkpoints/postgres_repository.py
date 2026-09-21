from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from ai_platform.agents.checkpoint import AgentExecutionCheckpoint
from app.control_plane.agent_runs.models import AgentRunStatus
from app.control_plane.agent_checkpoints.exceptions import (
    AgentCheckpointOwnershipLostError,
)
from app.control_plane.persistence.models import (
    AgentRunCheckpointRecord,
    AgentRunRecord,
)


class PostgreSQLAgentCheckpointsRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def save(
        self,
        checkpoint: AgentExecutionCheckpoint,
        *,
        lease_id: str | None = None,
        commit: bool = True,
    ) -> AgentExecutionCheckpoint:
        try:
            if lease_id is not None:
                run = self._session.execute(
                    select(AgentRunRecord)
                    .where(
                        AgentRunRecord.run_id == checkpoint.run_id,
                    )
                    .with_for_update()
                ).scalar_one_or_none()

                now = datetime.now(UTC)

                if (
                    run is None
                    or run.status != AgentRunStatus.RUNNING.value
                    or run.lease_id != lease_id
                    or run.lease_expires_at is None
                    or run.lease_expires_at <= now
                ):
                    raise AgentCheckpointOwnershipLostError(
                        "Cannot persist agent execution checkpoint because "
                        "the worker no longer owns the active run lease."
                    )

            record = AgentRunCheckpointRecord(
                run_id=checkpoint.run_id,
                agent_name=checkpoint.agent_name,
                session_id=checkpoint.session_id,
                user_id=checkpoint.user_id,
                schema_version=checkpoint.schema_version,
                position=checkpoint.position.value,
                tool_round=checkpoint.tool_round,
                checkpoint_payload=checkpoint.to_dict(),
            )

            self._session.add(record)
            self._session.flush()

            if commit:
                self._session.commit()

        except Exception:
            if commit:
                self._session.rollback()
            raise

        return checkpoint

    def get_latest(
        self,
        run_id: str,
    ) -> AgentExecutionCheckpoint | None:
        statement = (
            select(AgentRunCheckpointRecord)
            .where(AgentRunCheckpointRecord.run_id == run_id)
            .order_by(
                AgentRunCheckpointRecord.created_at.desc(),
                AgentRunCheckpointRecord.id.desc(),
            )
            .limit(1)
        )

        record = self._session.scalar(statement)

        if record is None:
            return None

        return AgentExecutionCheckpoint.from_dict(dict(record.checkpoint_payload))
