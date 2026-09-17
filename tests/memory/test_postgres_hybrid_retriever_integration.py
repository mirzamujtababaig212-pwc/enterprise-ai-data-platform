from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone

import pytest
from sqlalchemy import delete

from app.control_plane.persistence.database import SessionLocal
from app.control_plane.persistence.models import (
    MemoryEmbeddingRecord,
    MemoryItemRecord,
)
from memory.embeddings.postgres import PostgreSQLMemoryEmbeddingStore
from memory.models import MemoryItem
from memory.retrieval.hybrid import HybridMemoryRetriever
from memory.retrieval.postgres_lexical import PostgreSQLLexicalMemoryRetriever
from memory.retrieval.postgres_semantic import PostgreSQLSemanticMemoryRetriever
from rag.models import EmbeddingIdentity, EmbeddingResult

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_POSTGRES_INTEGRATION") != "1",
    reason="PostgreSQL integration tests require RUN_POSTGRES_INTEGRATION=1",
)


class FakeEmbeddingService:
    def __init__(
        self,
        vectors: dict[str, tuple[float, ...]],
        identity: EmbeddingIdentity,
    ) -> None:
        self.vectors = vectors
        self.identity = identity

    async def embed(self, text: str) -> tuple[float, ...]:
        return self.vectors[text]

    async def embed_with_metadata(self, text: str) -> EmbeddingResult:
        return EmbeddingResult(
            vector=self.vectors[text],
            identity=self.identity,
        )


def _memory(
    memory_id: str,
    content: str,
    *,
    namespace: str,
    memory_type: str = "semantic",
    created_at: datetime | None = None,
) -> MemoryItem:
    return MemoryItem(
        id=memory_id,
        memory_type=memory_type,  # type: ignore[arg-type]
        content=content,
        namespace=namespace,
        created_at=created_at or datetime.now(timezone.utc),
        metadata={"source": "postgres-hybrid-retrieval-test"},
    )


def _embedding(
    vector: tuple[float, ...],
) -> EmbeddingResult:
    identity = EmbeddingIdentity(
        requested_provider="test-provider",
        requested_model="test-embedding-model",
        resolved_provider="test-provider",
        resolved_model="test-embedding-model",
        dimension=len(vector),
    )
    return EmbeddingResult(
        vector=vector,
        identity=identity,
    )


async def _put_memory(
    memory: MemoryItem,
    embedding: EmbeddingResult,
) -> None:
    session = SessionLocal()
    try:
        session.add(
            MemoryItemRecord(
                id=memory.id,
                memory_type=memory.memory_type,
                content=memory.content,
                namespace=memory.namespace,
                created_at=memory.created_at,
                expires_at=memory.expires_at,
                memory_metadata=dict(memory.metadata),
            )
        )
        session.commit()
    finally:
        session.close()

    await PostgreSQLMemoryEmbeddingStore(SessionLocal).put(
        memory.id,
        embedding,
    )


async def _cleanup(memory_ids: list[str]) -> None:
    session = SessionLocal()
    try:
        session.execute(
            delete(MemoryEmbeddingRecord).where(MemoryEmbeddingRecord.memory_id.in_(memory_ids))
        )
        session.execute(delete(MemoryItemRecord).where(MemoryItemRecord.id.in_(memory_ids)))
        session.commit()
    finally:
        session.close()


def test_postgresql_hybrid_memory_retriever_fuses_semantic_and_lexical_results():
    async def run() -> None:
        namespace = "postgres-hybrid-memory-retrieval"

        memories = [
            _memory(
                "hybrid-shared",
                "Deployment configuration requires production approval.",
                namespace=namespace,
                created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            ),
            _memory(
                "hybrid-semantic",
                "Production release monitoring tracks deployment health.",
                namespace=namespace,
                created_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
            ),
            _memory(
                "hybrid-lexical",
                "Deployment configuration procedures require validation.",
                namespace=namespace,
                created_at=datetime(2026, 1, 3, tzinfo=timezone.utc),
            ),
            _memory(
                "hybrid-unrelated",
                "Invoice validation requires a purchase order.",
                namespace=namespace,
                created_at=datetime(2026, 1, 4, tzinfo=timezone.utc),
            ),
        ]

        embeddings = {
            "hybrid-shared": _embedding((1.0, 0.0, 0.0, 0.0)),
            "hybrid-semantic": _embedding((0.9, 0.4, 0.0, 0.0)),
            "hybrid-lexical": _embedding((0.2, 0.9, 0.0, 0.0)),
            "hybrid-unrelated": _embedding((0.0, 0.0, 1.0, 0.0)),
        }

        query_embedding = _embedding((1.0, 0.0, 0.0, 0.0))

        try:
            for memory in memories:
                await _put_memory(memory, embeddings[memory.id])

            embedding_service = FakeEmbeddingService(
                vectors={
                    "deployment configuration": query_embedding.vector,
                },
                identity=query_embedding.identity,
            )

            semantic_retriever = PostgreSQLSemanticMemoryRetriever(
                embedding_service=embedding_service,
                session_factory=SessionLocal,
            )
            lexical_retriever = PostgreSQLLexicalMemoryRetriever(
                session_factory=SessionLocal,
            )

            retriever = HybridMemoryRetriever(
                semantic_retriever=semantic_retriever,
                lexical_retriever=lexical_retriever,
                candidate_k=3,
                rrf_k=60,
            )

            results = await retriever.retrieve(
                "deployment configuration",
                namespace=namespace,
                top_k=3,
            )

            result_ids = [item.id for item in results]

            assert "hybrid-shared" in result_ids
            assert "hybrid-unrelated" not in result_ids
            assert len(result_ids) == 3
            assert len(result_ids) == len(set(result_ids))
        finally:
            await _cleanup([memory.id for memory in memories])

    asyncio.run(run())
