from __future__ import annotations

import asyncio
import math
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
from memory.evaluation.datasets.memory_retrieval_quality import (
    MEMORY_RETRIEVAL_QUALITY_CASES,
    MEMORY_RETRIEVAL_QUALITY_ITEMS,
)
from memory.evaluation.evaluator import MemoryRetrievalEvaluator
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


DIMENSION = 16


def _vector_for_text(text: str) -> tuple[float, ...]:
    """Deterministic synthetic concept embedding for benchmark isolation."""
    text_lower = text.lower()

    concepts = {
        "deployment": ("deployment", "deploy", "release", "rollout"),
        "configuration": ("configuration", "config", "checklist"),
        "approval": ("approval", "approved", "authorization"),
        "monitoring": ("monitor", "monitoring", "service health"),
        "invoice": ("invoice", "supplier", "purchase order"),
        "purchase": ("purchase order", "supplier commitment"),
        "vendor": ("vendor", "onboarding", "compliance"),
        "access": ("access", "permission", "privileged"),
        "security": ("security", "incident", "escalated"),
        "atlas": ("atlas", "project"),
        "finance": ("revenue", "financial", "quarterly"),
        "hiring": ("hiring", "candidate", "interview"),
        "validation": ("validated", "validation", "checklist"),
    }

    values = [0.0] * DIMENSION

    for index, (_, terms) in enumerate(concepts.items()):
        if any(term in text_lower for term in terms):
            values[index] = 1.0

    if not any(values):
        values[15] = 1.0

    norm = math.sqrt(sum(value * value for value in values))
    return tuple(value / norm for value in values)


class SyntheticBenchmarkEmbeddingService:
    def __init__(self, identity: EmbeddingIdentity) -> None:
        self.identity = identity

    async def embed(self, text: str) -> tuple[float, ...]:
        return _vector_for_text(text)

    async def embed_with_metadata(self, text: str) -> EmbeddingResult:
        return EmbeddingResult(
            vector=_vector_for_text(text),
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


def test_postgresql_hybrid_memory_retriever_meets_quality_baseline():
    async def run() -> None:
        identity = EmbeddingIdentity(
            requested_provider="test-provider",
            requested_model="postgres-hybrid-quality-test",
            resolved_provider="test-provider",
            resolved_model="postgres-hybrid-quality-test",
            dimension=DIMENSION,
        )
        embedding_service = SyntheticBenchmarkEmbeddingService(identity)

        try:
            for item in MEMORY_RETRIEVAL_QUALITY_ITEMS:
                await _put_memory(
                    item,
                    EmbeddingResult(
                        vector=_vector_for_text(item.content),
                        identity=identity,
                    ),
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
                candidate_k=5,
                rrf_k=60,
            )

            evaluator = MemoryRetrievalEvaluator(retriever, k=5)
            result = await evaluator.evaluate(MEMORY_RETRIEVAL_QUALITY_CASES)

            assert result.evaluated_queries == len(MEMORY_RETRIEVAL_QUALITY_CASES)
            assert result.successful_queries == len(MEMORY_RETRIEVAL_QUALITY_CASES)
            assert result.failed_queries == 0
            assert result.recall_at_k >= 0.95
            assert result.precision_at_k >= 0.45
            assert result.mrr >= 0.90
            assert result.ndcg_at_k >= 0.92
            assert result.mean_latency_ms >= 0.0

            print()
            print("PostgreSQL hybrid quality baseline:")
            print(f"recall@5    = {result.recall_at_k:.6f}")
            print(f"precision@5 = {result.precision_at_k:.6f}")
            print(f"mrr         = {result.mrr:.6f}")
            print(f"ndcg@5      = {result.ndcg_at_k:.6f}")
            print(f"latency_ms  = {result.mean_latency_ms:.3f}")
        finally:
            await _cleanup([item.id for item in MEMORY_RETRIEVAL_QUALITY_ITEMS])

    asyncio.run(run())
