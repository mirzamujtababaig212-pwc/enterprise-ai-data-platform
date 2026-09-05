from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.control_plane.persistence.models import UsageEventRecord
from app.control_plane.usage.models import UsageEvent


class PostgreSQLUsageRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def record(self, event: UsageEvent) -> UsageEvent:
        record = UsageEventRecord(
            event_id=event.event_id,
            request_id=event.request_id,
            timestamp=event.timestamp,
            capability=event.capability,
            provider=event.provider,
            model=event.model,
            tokens_in=event.tokens_in,
            tokens_out=event.tokens_out,
            estimated_cost=event.estimated_cost,
            latency_ms=event.latency_ms,
            status=event.status,
        )

        self._session.add(record)
        self._session.commit()

        return event

    def list(
        self,
        *,
        request_id: str | None = None,
        capability: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[UsageEvent]:
        statement = select(UsageEventRecord)

        if request_id is not None:
            statement = statement.where(UsageEventRecord.request_id == request_id)

        if capability is not None:
            statement = statement.where(UsageEventRecord.capability == capability)

        if status is not None:
            statement = statement.where(UsageEventRecord.status == status)

        statement = statement.order_by(UsageEventRecord.id.desc()).limit(limit)

        records = list(self._session.scalars(statement).all())
        records.reverse()

        return [self._to_domain(record) for record in records]

    def get(self, event_id: str) -> UsageEvent | None:
        record = self._session.scalar(
            select(UsageEventRecord).where(UsageEventRecord.event_id == event_id)
        )

        if record is None:
            return None

        return self._to_domain(record)

    def clear(self) -> None:
        self._session.query(UsageEventRecord).delete()
        self._session.commit()

    @staticmethod
    def _to_domain(record: UsageEventRecord) -> UsageEvent:
        return UsageEvent(
            event_id=record.event_id,
            request_id=record.request_id,
            timestamp=record.timestamp,
            capability=record.capability,
            provider=record.provider,
            model=record.model,
            tokens_in=record.tokens_in,
            tokens_out=record.tokens_out,
            estimated_cost=record.estimated_cost,
            latency_ms=record.latency_ms,
            status=record.status,
        )
