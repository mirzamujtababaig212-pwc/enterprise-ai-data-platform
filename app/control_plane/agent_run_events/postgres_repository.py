from __future__ import annotations

from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ai_platform.agents.observability import (
    AgentExecutionEvent,
    AgentExecutionEventType,
)
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

        event_metadata = dict(event.metadata)

        if event.attempt is not None:
            event_metadata["_event_attempt"] = event.attempt

        record = AgentRunEventRecord(
            event_id=str(uuid4()),
            run_id=event.run_id,
            event_type=event.event_type.value,
            agent_name=event.agent_name,
            session_id=event.session_id,
            user_id=event.user_id,
            principal=event.principal,
            tool_round=event.tool_round,
            tool_name=event.tool_name,
            call_id=event.call_id,
            step_id=event.step_id,
            step_index=event.step_index,
            step_name=event.step_name,
            provider=event.provider,
            model=event.model,
            event_metadata=event_metadata,
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
        event_type: AgentExecutionEventType | None = None,
        step_id: str | None = None,
        attempt: int | None = None,
        provider: str | None = None,
        cursor: str | None = None,
        limit: int = 100,
    ) -> list[AgentExecutionEvent]:
        statement = select(AgentRunEventRecord).where(
            AgentRunEventRecord.run_id == run_id,
        )

        if cursor is not None:
            from app.control_plane.agent_run_events.cursor import (
                decode_cursor,
            )

            cursor_created_at, cursor_id = decode_cursor(cursor)

            statement = statement.where(
                (
                    AgentRunEventRecord.created_at,
                    AgentRunEventRecord.id,
                )
                < (
                    cursor_created_at,
                    cursor_id,
                )
            )

        if event_type is not None:
            statement = statement.where(
                AgentRunEventRecord.event_type == event_type.value,
            )

        if step_id is not None:
            statement = statement.where(
                AgentRunEventRecord.step_id == step_id,
            )

        if provider is not None:
            statement = statement.where(
                AgentRunEventRecord.provider == provider,
            )

        statement = statement.order_by(
            AgentRunEventRecord.id.asc(),
        ).limit(limit + 1 if cursor is not None else limit)

        records = self._session.scalars(statement).all()

        events = [self._to_domain(record) for record in records]

        if attempt is not None:
            events = [event for event in events if event.attempt == attempt]

        return events

    @staticmethod
    def _to_domain(record: AgentRunEventRecord) -> AgentExecutionEvent:
        from ai_platform.agents.observability import AgentExecutionEventType

        metadata = dict(record.event_metadata)
        attempt = metadata.pop("_event_attempt", None)

        return AgentExecutionEvent(
            event_type=AgentExecutionEventType(record.event_type),
            agent_name=record.agent_name,
            run_id=record.run_id,
            session_id=record.session_id,
            user_id=record.user_id,
            principal=record.principal,
            tool_round=record.tool_round,
            tool_name=record.tool_name,
            call_id=record.call_id,
            step_id=record.step_id,
            step_index=record.step_index,
            step_name=record.step_name,
            attempt=attempt,
            provider=record.provider,
            model=record.model,
            metadata=metadata,
        )
