from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence

from sqlalchemy import func, select
from sqlalchemy.orm import Session
from sqlalchemy.sql import column
from sqlalchemy.dialects.postgresql import TSVECTOR

from app.control_plane.persistence.database import SessionLocal
from app.control_plane.persistence.models import MemoryItemRecord
from memory.models import MemoryItem, MemoryType


class PostgreSQLLexicalMemoryRetriever:
    """
    PostgreSQL-native lexical memory retriever.

    Namespace, memory-type, expiration, lexical matching, and ranking are
    performed by PostgreSQL. Results are converted
    directly into MemoryItem objects.
    """

    def __init__(
        self,
        session_factory: Callable[[], Session] = SessionLocal,
    ) -> None:
        self._session_factory = session_factory

    async def retrieve(
        self,
        query: str,
        *,
        namespace: str,
        memory_type: MemoryType | None = None,
        top_k: int = 5,
    ) -> Sequence[MemoryItem]:
        if not query.strip():
            raise ValueError("Query must not be empty.")

        if not namespace.strip():
            raise ValueError("Memory namespace must not be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        return await asyncio.to_thread(
            self._retrieve_sync,
            query,
            namespace,
            memory_type,
            top_k,
        )

    def _retrieve_sync(
        self,
        query: str,
        namespace: str,
        memory_type: MemoryType | None,
        top_k: int,
    ) -> tuple[MemoryItem, ...]:
        session = self._session_factory()

        try:
            content_tsv = column("content_tsv", TSVECTOR())
            tsquery = func.websearch_to_tsquery("simple", query)
            rank = func.ts_rank_cd(content_tsv, tsquery)

            now = func.now()

            statement = select(MemoryItemRecord, rank.label("rank")).where(
                MemoryItemRecord.namespace == namespace,
                (MemoryItemRecord.expires_at.is_(None) | (MemoryItemRecord.expires_at > now)),
                content_tsv.op("@@")(tsquery),
            )

            if memory_type is not None:
                statement = statement.where(
                    MemoryItemRecord.memory_type == memory_type,
                )

            statement = statement.order_by(
                rank.desc(),
                MemoryItemRecord.created_at.desc(),
                MemoryItemRecord.id.asc(),
            ).limit(top_k)

            rows = session.execute(statement).all()

            return tuple(self._to_memory_item(record) for record, _rank in rows)
        finally:
            session.close()

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
