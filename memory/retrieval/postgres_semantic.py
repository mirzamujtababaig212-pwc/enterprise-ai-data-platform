from __future__ import annotations

import asyncio
import math
from collections.abc import Callable, Sequence

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.control_plane.persistence.database import SessionLocal
from app.control_plane.persistence.models import (
    MemoryEmbeddingRecord,
    MemoryItemRecord,
)
from memory.models import MemoryItem, MemoryType
from memory.retrieval.contracts import MemoryRetrievalResult
from rag.compatibility import EmbeddingCompatibilityPolicy
from rag.contracts import EmbeddingService
from rag.models import EmbeddingIdentity, EmbeddingResult


class PostgreSQLSemanticMemoryRetriever:
    """
    PostgreSQL-backed semantic retrieval for memory items.

    Query embeddings are generated through the existing embedding service.
    Stored memory embeddings are searched using exact cosine distance.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        session_factory: Callable[[], Session] = SessionLocal,
    ) -> None:
        self._embedding_service = embedding_service
        self._session_factory = session_factory

    async def retrieve(
        self,
        query: str,
        *,
        namespace: str,
        memory_type: MemoryType | None = None,
        top_k: int = 10,
    ) -> Sequence[MemoryItem]:
        if not query.strip():
            raise ValueError("Query must not be empty.")

        if not namespace.strip():
            raise ValueError("Memory namespace must not be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        embedding = await self._embedding_service.embed_with_metadata(query)

        return await asyncio.to_thread(
            self._retrieve_sync,
            embedding,
            namespace,
            memory_type,
            top_k,
        )

    def _retrieve_sync(
        self,
        embedding: EmbeddingResult,
        namespace: str,
        memory_type: MemoryType | None,
        top_k: int,
    ) -> list[MemoryItem]:
        query_vector = [float(value) for value in embedding.vector]
        query_dimension = len(query_vector)

        if query_dimension == 0:
            return []

        if query_dimension != embedding.identity.dimension:
            raise ValueError(
                "Embedding vector dimension does not match " "embedding identity dimension."
            )

        query_norm = math.sqrt(sum(value * value for value in query_vector))

        session = self._session_factory()

        try:
            dimensions_statement = (
                select(MemoryEmbeddingRecord.embedding_dimension)
                .join(
                    MemoryItemRecord,
                    MemoryItemRecord.id == MemoryEmbeddingRecord.memory_id,
                )
                .where(
                    MemoryItemRecord.namespace == namespace,
                    MemoryEmbeddingRecord.embedding_dimension.is_not(None),
                    (
                        MemoryItemRecord.expires_at.is_(None)
                        | (MemoryItemRecord.expires_at > func.now())
                    ),
                )
            )

            if memory_type is not None:
                dimensions_statement = dimensions_statement.where(
                    MemoryItemRecord.memory_type == memory_type
                )

            stored_dimensions = {
                int(dimension)
                for dimension in session.scalars(dimensions_statement.distinct()).all()
                if dimension is not None
            }

            if stored_dimensions and query_dimension not in stored_dimensions:
                expected_dimensions = ", ".join(
                    str(dimension) for dimension in sorted(stored_dimensions)
                )
                raise ValueError(
                    "Embedding dimensions must match: "
                    f"expected {expected_dimensions}, got {query_dimension}."
                )

            distance = MemoryEmbeddingRecord.embedding.cosine_distance(query_vector)

            statement = (
                select(
                    MemoryItemRecord,
                    MemoryEmbeddingRecord.embedding,
                    MemoryEmbeddingRecord.embedding_dimension,
                    MemoryEmbeddingRecord.embedding_requested_provider,
                    MemoryEmbeddingRecord.embedding_requested_model,
                    MemoryEmbeddingRecord.embedding_resolved_provider,
                    MemoryEmbeddingRecord.embedding_resolved_model,
                    distance.label("distance"),
                )
                .join(
                    MemoryEmbeddingRecord,
                    MemoryEmbeddingRecord.memory_id == MemoryItemRecord.id,
                )
                .where(
                    MemoryItemRecord.namespace == namespace,
                    MemoryEmbeddingRecord.embedding_dimension == query_dimension,
                    (
                        MemoryItemRecord.expires_at.is_(None)
                        | (MemoryItemRecord.expires_at > func.now())
                    ),
                )
            )

            if memory_type is not None:
                statement = statement.where(MemoryItemRecord.memory_type == memory_type)

            if query_norm == 0.0:
                statement = statement.order_by(
                    MemoryItemRecord.created_at.desc(),
                    MemoryItemRecord.id.asc(),
                ).limit(top_k)

                rows = session.execute(statement).all()

                return [
                    MemoryRetrievalResult(
                        item=self._validated_memory_item(
                            record,
                            embedding,
                            stored_dimension,
                            stored_requested_provider,
                            stored_requested_model,
                            stored_resolved_provider,
                            stored_resolved_model,
                        ),
                        retrieval_method="postgresql.semantic.cosine",
                        rank=rank,
                        retrieval_score=None,
                        provenance={"ordering": "created_at_desc_id_asc"},
                    )
                    for rank, (
                        record,
                        _,
                        stored_dimension,
                        stored_requested_provider,
                        stored_requested_model,
                        stored_resolved_provider,
                        stored_resolved_model,
                        _distance,
                    ) in enumerate(rows, start=1)
                ]

            statement = statement.order_by(
                distance.asc(),
                MemoryItemRecord.id.asc(),
            ).limit(top_k)

            rows = session.execute(statement).all()

            return [
                MemoryRetrievalResult(
                    item=self._validated_memory_item(
                        record,
                        embedding,
                        stored_dimension,
                        stored_requested_provider,
                        stored_requested_model,
                        stored_resolved_provider,
                        stored_resolved_model,
                    ),
                    retrieval_method="postgresql.semantic.cosine",
                    rank=rank,
                    retrieval_score=float(1.0 - distance_value),
                )
                for rank, (
                    record,
                    _,
                    stored_dimension,
                    stored_requested_provider,
                    stored_requested_model,
                    stored_resolved_provider,
                    stored_resolved_model,
                    distance_value,
                ) in enumerate(rows, start=1)
            ]

        finally:
            session.close()

    @staticmethod
    def _validated_memory_item(
        record: MemoryItemRecord,
        query_embedding: EmbeddingResult,
        stored_dimension: int,
        stored_requested_provider: str | None,
        stored_requested_model: str,
        stored_resolved_provider: str,
        stored_resolved_model: str,
    ) -> MemoryItem:
        stored_identity = EmbeddingIdentity(
            requested_provider=stored_requested_provider,
            requested_model=stored_requested_model,
            resolved_provider=stored_resolved_provider,
            resolved_model=stored_resolved_model,
            dimension=stored_dimension,
        )

        EmbeddingCompatibilityPolicy.validate(
            query=query_embedding.identity,
            stored=stored_identity,
        )

        return PostgreSQLSemanticMemoryRetriever._to_memory_item(record)

    @staticmethod
    def _to_memory_item(record: MemoryItemRecord) -> MemoryItem:
        return MemoryItem(
            id=record.id,
            memory_type=record.memory_type,  # type: ignore[arg-type]
            content=record.content,
            namespace=record.namespace,
            created_at=record.created_at,
            expires_at=record.expires_at,
            metadata=dict(record.memory_metadata),
        )
