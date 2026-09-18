from __future__ import annotations

from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ai_platform.agents.observability import AgentExecutionEvent
from app.control_plane.persistence.models import AgentRunEventRecord


class PostgreSQLAgentRunEventsRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def record(
        self,
        event: AgentExecutionEvent,
        *,
        commit: bool = True,
    ) -> AgentExecutionEvent:
        if event.run_id is None:
            return event

        record = AgentRunEventRecord(
            event_id=str(uuid4()),
            run_id=event.run_id,
            event_type=event.event_type.value,
            agent_name=event.agent_name,
            session_id=event.session_id,
            tool_round=event.tool_round,
            tool_name=event.tool_name,
            call_id=event.call_id,
            provider=event.provider,
            model=event.model,
            event_metadata=dict(event.metadata),
        )

        try:
            self._session.add(record)
            self._session.flush()

            if commit:
                self._session.commit()
        except IntegrityError:
            if commit:
                self._session.rollback()
            raise
        except Exception:
            if commit:
                self._session.rollback()
            raise

        return event

    def list(
        self,
        run_id: str,
        *,
        limit: int = 100,
    ) -> list[AgentExecutionEvent]:
        statement = (
            select(AgentRunEventRecord)
            .where(AgentRunEventRecord.run_id == run_id)
            .order_by(AgentRunEventRecord.id.asc())
            .limit(limit)
        )

        records = self._session.scalars(statement).all()

        return [self._to_domain(record) for record in records]

    @staticmethod
    def _to_domain(record: AgentRunEventRecord) -> AgentExecutionEvent:
        from ai_platform.agents.observability import AgentExecutionEventType

        return AgentExecutionEvent(
            event_type=AgentExecutionEventType(record.event_type),
            agent_name=record.agent_name,
            run_id=record.run_id,
            session_id=record.session_id,
            tool_round=record.tool_round,
            tool_name=record.tool_name,
            call_id=record.call_id,
            provider=record.provider,
            model=record.model,
            metadata=dict(record.event_metadata),
        )
