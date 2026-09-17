from __future__ import annotations

import asyncio
from collections.abc import Callable

from sqlalchemy.orm import Session

from app.control_plane.persistence.database import SessionLocal
from app.control_plane.persistence.models import MemoryEmbeddingRecord
from rag.models import EmbeddingIdentity, EmbeddingResult


class PostgreSQLMemoryEmbeddingStore:
    """
    PostgreSQL-backed persistence for memory embeddings.

    Embedding provenance is stored alongside the vector so that semantic
    retrieval can validate compatibility with the query embedding.
    """

    def __init__(
        self,
        session_factory: Callable[[], Session] = SessionLocal,
    ) -> None:
        self._session_factory = session_factory

    async def put(
        self,
        memory_id: str,
        embedding: EmbeddingResult,
    ) -> None:
        if not memory_id.strip():
            raise ValueError("Memory ID must not be empty.")

        await asyncio.to_thread(
            self._put_sync,
            memory_id,
            embedding,
        )

    async def get(
        self,
        memory_id: str,
    ) -> EmbeddingResult | None:
        if not memory_id.strip():
            raise ValueError("Memory ID must not be empty.")

        return await asyncio.to_thread(
            self._get_sync,
            memory_id,
        )

    async def delete(
        self,
        memory_id: str,
    ) -> None:
        if not memory_id.strip():
            raise ValueError("Memory ID must not be empty.")

        await asyncio.to_thread(
            self._delete_sync,
            memory_id,
        )

    def _put_sync(
        self,
        memory_id: str,
        embedding: EmbeddingResult,
    ) -> None:
        identity = embedding.identity

        if len(embedding.vector) != identity.dimension:
            raise ValueError(
                "Embedding vector dimension does not match " "embedding identity dimension."
            )

        session = self._session_factory()

        try:
            record = session.get(
                MemoryEmbeddingRecord,
                memory_id,
            )

            if record is None:
                record = MemoryEmbeddingRecord(
                    memory_id=memory_id,
                    embedding=list(embedding.vector),
                    embedding_dimension=identity.dimension,
                    embedding_requested_provider=identity.requested_provider,
                    embedding_requested_model=identity.requested_model,
                    embedding_resolved_provider=identity.resolved_provider,
                    embedding_resolved_model=identity.resolved_model,
                )
                session.add(record)
            else:
                record.embedding = list(embedding.vector)
                record.embedding_dimension = identity.dimension
                record.embedding_requested_provider = identity.requested_provider
                record.embedding_requested_model = identity.requested_model
                record.embedding_resolved_provider = identity.resolved_provider
                record.embedding_resolved_model = identity.resolved_model

            session.commit()

        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def _get_sync(
        self,
        memory_id: str,
    ) -> EmbeddingResult | None:
        session = self._session_factory()

        try:
            record = session.get(
                MemoryEmbeddingRecord,
                memory_id,
            )

            if record is None:
                return None

            identity = EmbeddingIdentity(
                requested_provider=record.embedding_requested_provider,
                requested_model=record.embedding_requested_model,
                resolved_provider=record.embedding_resolved_provider,
                resolved_model=record.embedding_resolved_model,
                dimension=record.embedding_dimension,
            )

            return EmbeddingResult(
                vector=tuple(record.embedding),
                identity=identity,
            )
        finally:
            session.close()

    def _delete_sync(
        self,
        memory_id: str,
    ) -> None:
        session = self._session_factory()

        try:
            record = session.get(
                MemoryEmbeddingRecord,
                memory_id,
            )

            if record is not None:
                session.delete(record)
                session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
