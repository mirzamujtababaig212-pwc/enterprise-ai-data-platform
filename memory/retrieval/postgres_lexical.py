from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import Session
from sqlalchemy.sql import column

from app.control_plane.persistence.database import SessionLocal
from app.control_plane.persistence.models import MemoryItemRecord
from memory.models import MemoryItem, MemoryType
from memory.retrieval.contracts import MemoryRetrievalResult


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
    ) -> Sequence[MemoryRetrievalResult]:
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
    ) -> tuple[MemoryRetrievalResult, ...]:
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

            return tuple(
                MemoryRetrievalResult(
                    item=self._to_memory_item(record),
                    retrieval_method="postgresql.lexical",
                    rank=rank,
                    retrieval_score=float(score),
                )
                for rank, (record, score) in enumerate(rows, start=1)
            )
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
