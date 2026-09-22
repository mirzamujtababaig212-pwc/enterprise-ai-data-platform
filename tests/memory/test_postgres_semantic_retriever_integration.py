from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import delete

from app.control_plane.persistence.database import SessionLocal
from app.control_plane.persistence.models import (
    MemoryEmbeddingRecord,
    MemoryItemRecord,
)
from memory.embeddings.postgres import PostgreSQLMemoryEmbeddingStore
from memory.models import MemoryItem
from memory.retrieval.postgres_semantic import PostgreSQLSemanticMemoryRetriever
from rag.models import EmbeddingIdentity, EmbeddingResult

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_POSTGRES_INTEGRATION") != "1",
    reason="PostgreSQL integration tests require RUN_POSTGRES_INTEGRATION=1",
)


@dataclass
class FakeEmbeddingService:
    vectors: dict[str, tuple[float, ...]]
    identity: EmbeddingIdentity

    async def embed(self, text: str) -> tuple[float, ...]:
        return self.vectors[text]

    async def embed_with_metadata(self, text: str) -> EmbeddingResult:
        return EmbeddingResult(
            vector=self.vectors[text],
            identity=self.identity,
        )


def _session_factory():
    engine = SessionLocal.kw["bind"]
    return engine, SessionLocal


def _memory(
    memory_id: str,
    content: str,
    *,
    namespace: str = "semantic-retrieval-test",
    memory_type: str = "semantic",
    created_at: datetime | None = None,
    expires_at: datetime | None = None,
) -> MemoryItem:
    return MemoryItem(
        id=memory_id,
        memory_type=memory_type,  # type: ignore[arg-type]
        content=content,
        namespace=namespace,
        created_at=created_at or datetime.now(UTC),
        expires_at=expires_at,
        metadata={"source": "semantic-retrieval-test"},
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


def _embedding(
    vector: tuple[float, ...],
    *,
    dimension: int = 4,
) -> EmbeddingResult:
    identity = EmbeddingIdentity(
        requested_provider="test-provider",
        requested_model="test-embedding-model",
        resolved_provider="test-provider",
        resolved_model="test-embedding-model",
        dimension=dimension,
    )
    return EmbeddingResult(
        vector=vector,
        identity=identity,
    )


def test_postgresql_semantic_memory_retriever_returns_ranked_matches():
    async def run():
        memories = [
            _memory(
                "semantic-memory-1",
                "Deployment configuration requires production approval.",
            ),
            _memory(
                "semantic-memory-2",
                "Deployment monitoring tracks production health.",
            ),
            _memory(
                "semantic-memory-3",
                "Invoice validation requires a purchase order.",
            ),
        ]

        embeddings = [
            _embedding((1.0, 0.0, 0.0, 0.0)),
            _embedding((0.8, 0.6, 0.0, 0.0)),
            _embedding((0.0, 0.0, 1.0, 0.0)),
        ]

        try:
            for memory, embedding in zip(memories, embeddings, strict=True):
                await _put_memory(memory, embedding)

            service = FakeEmbeddingService(
                vectors={
                    "deployment query": (1.0, 0.0, 0.0, 0.0),
                },
                identity=embeddings[0].identity,
            )

            retriever = PostgreSQLSemanticMemoryRetriever(
                embedding_service=service,
                session_factory=SessionLocal,
            )

            results = await retriever.retrieve(
                "deployment query",
                namespace="semantic-retrieval-test",
                top_k=3,
            )

            assert [result.item.id for result in results] == [
                "semantic-memory-1",
                "semantic-memory-2",
                "semantic-memory-3",
            ]
            assert [result.rank for result in results] == [1, 2, 3]
            assert results[0].retrieval_method == "postgresql.semantic.cosine"
            assert results[0].retrieval_score == pytest.approx(1.0)
            assert results[1].retrieval_score == pytest.approx(0.8)
            assert results[2].retrieval_score == pytest.approx(0.0)
            assert all(isinstance(result.item, MemoryItem) for result in results)
        finally:
            await _cleanup([memory.id for memory in memories])

    asyncio.run(run())


def test_postgresql_semantic_memory_retriever_filters_namespace_and_type():
    async def run():
        memories = [
            _memory(
                "semantic-filter-namespace",
                "Deployment configuration.",
            ),
            _memory(
                "semantic-filter-type",
                "Deployment configuration.",
                memory_type="episodic",
            ),
            _memory(
                "semantic-filter-other",
                "Deployment configuration.",
                namespace="other-namespace",
            ),
        ]

        embedding = _embedding((1.0, 0.0, 0.0, 0.0))

        try:
            for memory in memories:
                await _put_memory(memory, embedding)

            service = FakeEmbeddingService(
                vectors={"deployment": (1.0, 0.0, 0.0, 0.0)},
                identity=embedding.identity,
            )

            retriever = PostgreSQLSemanticMemoryRetriever(
                embedding_service=service,
                session_factory=SessionLocal,
            )

            results = await retriever.retrieve(
                "deployment",
                namespace="semantic-retrieval-test",
                memory_type="semantic",
                top_k=10,
            )

            assert [result.item.id for result in results] == ["semantic-filter-namespace"]
        finally:
            await _cleanup([memory.id for memory in memories])

    asyncio.run(run())


def test_postgresql_semantic_memory_retriever_excludes_expired_memory():
    async def run():
        now = datetime.now(UTC)

        active = _memory(
            "semantic-active",
            "Active deployment configuration.",
            created_at=now - timedelta(minutes=2),
        )
        expired = _memory(
            "semantic-expired",
            "Expired deployment configuration.",
            created_at=now - timedelta(minutes=1),
            expires_at=now - timedelta(seconds=1),
        )

        embedding = _embedding((1.0, 0.0, 0.0, 0.0))

        try:
            await _put_memory(active, embedding)
            await _put_memory(expired, embedding)

            service = FakeEmbeddingService(
                vectors={"deployment": (1.0, 0.0, 0.0, 0.0)},
                identity=embedding.identity,
            )

            retriever = PostgreSQLSemanticMemoryRetriever(
                embedding_service=service,
                session_factory=SessionLocal,
            )

            results = await retriever.retrieve(
                "deployment",
                namespace="semantic-retrieval-test",
                top_k=10,
            )

            assert [result.item.id for result in results] == ["semantic-active"]
        finally:
            await _cleanup([active.id, expired.id])

    asyncio.run(run())


def test_postgresql_semantic_memory_retriever_validates_arguments():
    async def run():
        service = FakeEmbeddingService(
            vectors={},
            identity=_embedding((1.0, 0.0, 0.0, 0.0)).identity,
        )

        retriever = PostgreSQLSemanticMemoryRetriever(
            embedding_service=service,
            session_factory=SessionLocal,
        )

        with pytest.raises(ValueError, match="Query must not be empty"):
            await retriever.retrieve(
                "",
                namespace="semantic-retrieval-test",
            )

        with pytest.raises(ValueError, match="namespace"):
            await retriever.retrieve(
                "query",
                namespace="",
            )

        with pytest.raises(ValueError, match="top_k"):
            await retriever.retrieve(
                "query",
                namespace="semantic-retrieval-test",
                top_k=0,
            )

    asyncio.run(run())


def test_postgresql_semantic_memory_retriever_rejects_incompatible_embedding_model():
    async def run():
        memory = _memory(
            "semantic-incompatible-model",
            "Deployment configuration requires approval.",
        )

        stored_embedding = _embedding(
            (1.0, 0.0, 0.0, 0.0),
        )

        incompatible_identity = EmbeddingIdentity(
            requested_provider="test-provider",
            requested_model="different-model",
            resolved_provider="test-provider",
            resolved_model="different-model",
            dimension=4,
        )

        query_embedding = EmbeddingResult(
            vector=(1.0, 0.0, 0.0, 0.0),
            identity=incompatible_identity,
        )

        try:
            await _put_memory(memory, stored_embedding)

            service = FakeEmbeddingService(
                vectors={
                    "deployment": query_embedding.vector,
                },
                identity=query_embedding.identity,
            )

            retriever = PostgreSQLSemanticMemoryRetriever(
                embedding_service=service,
                session_factory=SessionLocal,
            )

            with pytest.raises(ValueError, match="models"):
                await retriever.retrieve(
                    "deployment",
                    namespace="semantic-retrieval-test",
                    top_k=10,
                )
        finally:
            await _cleanup([memory.id])

    asyncio.run(run())


def test_postgresql_semantic_memory_retriever_rejects_incompatible_embedding_provider():
    async def run():
        memory = _memory(
            "semantic-incompatible-provider",
            "Deployment configuration requires approval.",
        )

        stored_embedding = _embedding(
            (1.0, 0.0, 0.0, 0.0),
        )

        incompatible_identity = EmbeddingIdentity(
            requested_provider="different-provider",
            requested_model="test-embedding-model",
            resolved_provider="different-provider",
            resolved_model="test-embedding-model",
            dimension=4,
        )

        query_embedding = EmbeddingResult(
            vector=(1.0, 0.0, 0.0, 0.0),
            identity=incompatible_identity,
        )

        try:
            await _put_memory(memory, stored_embedding)

            service = FakeEmbeddingService(
                vectors={
                    "deployment": query_embedding.vector,
                },
                identity=query_embedding.identity,
            )

            retriever = PostgreSQLSemanticMemoryRetriever(
                embedding_service=service,
                session_factory=SessionLocal,
            )

            with pytest.raises(ValueError, match="providers"):
                await retriever.retrieve(
                    "deployment",
                    namespace="semantic-retrieval-test",
                    top_k=10,
                )
        finally:
            await _cleanup([memory.id])

    asyncio.run(run())
