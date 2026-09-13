from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.control_plane.persistence.database import SessionLocal
from app.control_plane.persistence.models import RAGChunkRecord
from rag.models import (
    DocumentChunk,
    EmbeddedChunk,
    EmbeddingIdentity,
    RetrievalResult,
)


class PostgreSQLVectorStore:
    """
    PostgreSQL/pgvector-backed implementation of the RAG VectorStore contract.

    The existing rag_chunks table remains the authoritative persistence model
    for chunk content, metadata, embedding provenance, and the vector itself.
    """

    def __init__(
        self,
        session_factory: Callable[[], Session] = SessionLocal,
    ) -> None:
        self._session_factory = session_factory

    async def close(self) -> None:
        return None

    async def upsert(
        self,
        chunks: Sequence[EmbeddedChunk],
    ) -> None:
        if not chunks:
            return

        dimension = self._validate_batch_dimensions(chunks)

        await asyncio.to_thread(
            self._upsert_sync,
            chunks,
            dimension,
        )

    async def delete_chunks(
        self,
        chunk_ids: Sequence[str],
    ) -> None:
        ids = [chunk_id for chunk_id in chunk_ids if chunk_id]

        if not ids:
            return

        await asyncio.to_thread(self._delete_sync, ids)

    async def search(
        self,
        embedding: Sequence[float],
        top_k: int = 5,
        metadata_filter: Mapping[str, object] | None = None,
    ) -> Sequence[RetrievalResult]:
        if top_k <= 0:
            return []

        query = [float(value) for value in embedding]

        if not query:
            return []

        return await asyncio.to_thread(
            self._search_sync,
            query,
            top_k,
            metadata_filter,
        )

    def _upsert_sync(
        self,
        chunks: Sequence[EmbeddedChunk],
        dimension: int,
    ) -> None:
        session: Session = self._session_factory()

        try:
            for item in chunks:
                identity = item.embedding_identity

                existing = session.scalar(
                    select(RAGChunkRecord).where(RAGChunkRecord.chunk_id == item.chunk.id)
                )

                if existing is None:
                    record = RAGChunkRecord(
                        chunk_id=item.chunk.id,
                        document_id=item.chunk.document_id,
                        chunk_index=item.chunk.chunk_index,
                        content=item.chunk.content,
                        chunk_metadata=dict(item.chunk.metadata),
                    )
                    session.add(record)
                else:
                    record = existing
                    record.document_id = item.chunk.document_id
                    record.chunk_index = item.chunk.chunk_index
                    record.content = item.chunk.content
                    record.chunk_metadata = dict(item.chunk.metadata)

                record.embedding = list(item.embedding)
                record.embedding_dimension = dimension
                record.embedding_model = identity.resolved_model if identity is not None else None
                record.embedding_requested_provider = (
                    identity.requested_provider if identity is not None else None
                )
                record.embedding_requested_model = (
                    identity.requested_model if identity is not None else None
                )
                record.embedding_resolved_provider = (
                    identity.resolved_provider if identity is not None else None
                )
                record.embedding_resolved_model = (
                    identity.resolved_model if identity is not None else None
                )

            session.commit()

        except Exception:
            session.rollback()
            raise

        finally:
            session.close()

    def _delete_sync(self, chunk_ids: Sequence[str]) -> None:
        session: Session = self._session_factory()

        try:
            session.execute(delete(RAGChunkRecord).where(RAGChunkRecord.chunk_id.in_(chunk_ids)))
            session.commit()

        except Exception:
            session.rollback()
            raise

        finally:
            session.close()

    def _search_sync(
        self,
        query: list[float],
        top_k: int,
        metadata_filter: Mapping[str, object] | None,
    ) -> list[RetrievalResult]:
        session: Session = self._session_factory()

        try:
            distance = RAGChunkRecord.embedding.cosine_distance(query)

            statement = select(RAGChunkRecord, distance.label("distance")).where(
                RAGChunkRecord.embedding.is_not(None),
                RAGChunkRecord.embedding_dimension == len(query),
            )

            if metadata_filter:
                for key, value in metadata_filter.items():
                    statement = statement.where(
                        RAGChunkRecord.chunk_metadata[key].as_string() == str(value)
                    )

            statement = statement.order_by(
                distance.asc(),
                RAGChunkRecord.chunk_id.asc(),
            ).limit(top_k)

            rows = session.execute(statement).all()

            results: list[RetrievalResult] = []

            for record, distance_value in rows:
                results.append(
                    RetrievalResult(
                        chunk=DocumentChunk(
                            id=record.chunk_id,
                            document_id=record.document_id,
                            content=record.content,
                            metadata=dict(record.chunk_metadata),
                            chunk_index=record.chunk_index,
                        ),
                        score=1.0 - float(distance_value),
                        embedding_identity=self._embedding_identity(record),
                    )
                )

            return results

        finally:
            session.close()

    @staticmethod
    def _validate_batch_dimensions(
        chunks: Sequence[EmbeddedChunk],
    ) -> int:
        dimension = len(chunks[0].embedding)

        if dimension == 0:
            raise ValueError("Embeddings must not be empty.")

        for item in chunks:
            if len(item.embedding) != dimension:
                raise ValueError("All embeddings must have the same dimension.")

        return dimension

    @staticmethod
    def _embedding_identity(
        record: RAGChunkRecord,
    ) -> EmbeddingIdentity | None:
        if (
            record.embedding_dimension is None
            or record.embedding_requested_model is None
            or record.embedding_resolved_provider is None
            or record.embedding_resolved_model is None
        ):
            return None

        return EmbeddingIdentity(
            requested_provider=record.embedding_requested_provider,
            requested_model=record.embedding_requested_model,
            resolved_provider=record.embedding_resolved_provider,
            resolved_model=record.embedding_resolved_model,
            dimension=record.embedding_dimension,
        )
