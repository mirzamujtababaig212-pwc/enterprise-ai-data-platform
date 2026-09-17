from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from datetime import datetime, timezone

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.control_plane.persistence.database import SessionLocal
from app.control_plane.persistence.models import MemoryItemRecord
from memory.models import MemoryItem, MemoryType


class PostgreSQLMemoryStore:
    """
    PostgreSQL-backed implementation of the MemoryStore contract.

    Persistence semantics mirror InMemoryMemoryStore: namespace/type
    filtering, newest-first ordering, and expiration handling.
    """

    def __init__(
        self,
        session_factory: Callable[[], Session] = SessionLocal,
    ) -> None:
        self._session_factory = session_factory

    async def put(self, item: MemoryItem) -> None:
        await asyncio.to_thread(self._put_sync, item)

    async def get(self, memory_id: str) -> MemoryItem | None:
        return await asyncio.to_thread(self._get_sync, memory_id)

    async def search(
        self,
        namespace: str,
        *,
        memory_type: MemoryType | None = None,
        limit: int = 10,
    ) -> Sequence[MemoryItem]:
        if limit <= 0:
            return []

        return await asyncio.to_thread(
            self._search_sync,
            namespace,
            memory_type,
            limit,
        )

    async def delete(self, memory_id: str) -> None:
        await asyncio.to_thread(self._delete_sync, memory_id)

    def _put_sync(self, item: MemoryItem) -> None:
        session = self._session_factory()

        try:
            record = session.get(MemoryItemRecord, item.id)

            if record is None:
                record = MemoryItemRecord(id=item.id)
                session.add(record)

            record.memory_type = item.memory_type
            record.content = item.content
            record.namespace = item.namespace
            record.created_at = item.created_at
            record.expires_at = item.expires_at
            record.memory_metadata = dict(item.metadata)

            session.commit()

        except Exception:
            session.rollback()
            raise

        finally:
            session.close()

    def _get_sync(self, memory_id: str) -> MemoryItem | None:
        session = self._session_factory()

        try:
            record = session.get(MemoryItemRecord, memory_id)

            if record is None or self._is_expired(record):
                return None

            return self._to_memory_item(record)

        finally:
            session.close()

    def _search_sync(
        self,
        namespace: str,
        memory_type: MemoryType | None,
        limit: int,
    ) -> list[MemoryItem]:
        session = self._session_factory()

        try:
            now = datetime.now(timezone.utc)

            statement = select(MemoryItemRecord).where(
                MemoryItemRecord.namespace == namespace,
                (MemoryItemRecord.expires_at.is_(None) | (MemoryItemRecord.expires_at > now)),
            )

            if memory_type is not None:
                statement = statement.where(
                    MemoryItemRecord.memory_type == memory_type,
                )

            records = session.scalars(
                statement.order_by(
                    MemoryItemRecord.created_at.desc(),
                    MemoryItemRecord.id.asc(),
                ).limit(limit)
            ).all()

            return [self._to_memory_item(record) for record in records]

        finally:
            session.close()

    def _delete_sync(self, memory_id: str) -> None:
        session = self._session_factory()

        try:
            session.execute(
                delete(MemoryItemRecord).where(
                    MemoryItemRecord.id == memory_id,
                )
            )
            session.commit()

        except Exception:
            session.rollback()
            raise

        finally:
            session.close()

    @staticmethod
    def _is_expired(record: MemoryItemRecord) -> bool:
        if record.expires_at is None:
            return False

        expires_at = record.expires_at

        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)

        return expires_at <= datetime.now(timezone.utc)

    @staticmethod
    def _to_memory_item(record: MemoryItemRecord) -> MemoryItem:
        return MemoryItem(
            id=record.id,
            memory_type=record.memory_type,
            content=record.content,
            namespace=record.namespace,
            created_at=record.created_at,
            metadata=dict(record.memory_metadata),
            expires_at=record.expires_at,
        )
