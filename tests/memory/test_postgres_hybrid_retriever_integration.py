from __future__ import annotations

import asyncio
import math
import os
from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine, delete, text
from sqlalchemy.orm import sessionmaker

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
from memory.retrieval.reranker import CrossEncoderMemoryReranker
from memory.retrieval.reranking import RerankingMemoryRetriever
from rag.models import EmbeddingIdentity, EmbeddingResult

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_POSTGRES_INTEGRATION") != "1",
    reason="PostgreSQL integration tests require RUN_POSTGRES_INTEGRATION=1",
)


def _postgres_session_factory():
    host = os.getenv("POSTGRES_TEST_HOST", "localhost")
    port = os.getenv("POSTGRES_TEST_PORT", "5432")
    user = os.getenv("POSTGRES_TEST_USER", "postgres")
    password = os.getenv("POSTGRES_TEST_PASSWORD", "postgres")
    database = os.getenv("POSTGRES_TEST_DB", "vehicle_platform")

    engine = create_engine(
        f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}",
        pool_pre_ping=True,
        future=True,
    )

    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))

    return (
        sessionmaker(
            bind=engine,
            autoflush=False,
            autocommit=False,
            expire_on_commit=False,
        ),
        engine,
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
        created_at=created_at or datetime.now(UTC),
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
    session_factory,
) -> None:
    session = session_factory()
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

    await PostgreSQLMemoryEmbeddingStore(session_factory).put(
        memory.id,
        embedding,
    )


async def _cleanup(memory_ids: list[str], session_factory) -> None:
    session = session_factory()
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
        session_factory, engine = _postgres_session_factory()
        namespace = "postgres-hybrid-memory-retrieval"

        memories = [
            _memory(
                "hybrid-shared",
                "Deployment configuration requires production approval.",
                namespace=namespace,
                created_at=datetime(2026, 1, 1, tzinfo=UTC),
            ),
            _memory(
                "hybrid-semantic",
                "Production release monitoring tracks deployment health.",
                namespace=namespace,
                created_at=datetime(2026, 1, 2, tzinfo=UTC),
            ),
            _memory(
                "hybrid-lexical",
                "Deployment configuration procedures require validation.",
                namespace=namespace,
                created_at=datetime(2026, 1, 3, tzinfo=UTC),
            ),
            _memory(
                "hybrid-unrelated",
                "Invoice validation requires a purchase order.",
                namespace=namespace,
                created_at=datetime(2026, 1, 4, tzinfo=UTC),
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
                await _put_memory(memory, embeddings[memory.id], session_factory)

            embedding_service = FakeEmbeddingService(
                vectors={
                    "deployment configuration": query_embedding.vector,
                },
                identity=query_embedding.identity,
            )

            semantic_retriever = PostgreSQLSemanticMemoryRetriever(
                embedding_service=embedding_service,
                session_factory=session_factory,
            )
            lexical_retriever = PostgreSQLLexicalMemoryRetriever(
                session_factory=session_factory,
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

            result_ids = [result.item.id for result in results]

            assert "hybrid-shared" in result_ids
            assert "hybrid-unrelated" not in result_ids
            assert len(result_ids) == 3
            assert len(result_ids) == len(set(result_ids))
        finally:
            await _cleanup([memory.id for memory in memories], session_factory)
            engine.dispose()

    asyncio.run(run())


def test_postgresql_hybrid_memory_retriever_meets_quality_baseline():
    async def run() -> None:
        session_factory, engine = _postgres_session_factory()
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
                    session_factory,
                )

            semantic_retriever = PostgreSQLSemanticMemoryRetriever(
                embedding_service=embedding_service,
                session_factory=session_factory,
            )
            lexical_retriever = PostgreSQLLexicalMemoryRetriever(
                session_factory=session_factory,
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
            await _cleanup(
                [item.id for item in MEMORY_RETRIEVAL_QUALITY_ITEMS],
                session_factory,
            )
            engine.dispose()

    asyncio.run(run())


def test_postgresql_hybrid_cross_encoder_memory_retriever_quality():
    """Compare experimental cross-encoder reranking with hybrid retrieval.

    This benchmark is observational rather than a production-quality gate:
    the reranker must execute successfully and produce measurable metrics,
    but it is not required to improve the hybrid baseline.
    """

    async def run() -> None:
        session_factory, engine = _postgres_session_factory()
        identity = EmbeddingIdentity(
            requested_provider="test-provider",
            requested_model="postgres-hybrid-reranker-quality-test",
            resolved_provider="test-provider",
            resolved_model="postgres-hybrid-reranker-quality-test",
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
                    session_factory,
                )

            semantic_retriever = PostgreSQLSemanticMemoryRetriever(
                embedding_service=embedding_service,
                session_factory=session_factory,
            )
            lexical_retriever = PostgreSQLLexicalMemoryRetriever(
                session_factory=session_factory,
            )

            baseline_retriever = HybridMemoryRetriever(
                semantic_retriever=semantic_retriever,
                lexical_retriever=lexical_retriever,
                candidate_k=20,
                rrf_k=60,
            )

            reranked_hybrid_retriever = HybridMemoryRetriever(
                semantic_retriever=semantic_retriever,
                lexical_retriever=lexical_retriever,
                candidate_k=20,
                rrf_k=60,
            )

            reranker = CrossEncoderMemoryReranker(
                model_id=CrossEncoderMemoryReranker.DEFAULT_MODEL_ID,
                onnx_filename=CrossEncoderMemoryReranker.DEFAULT_ONNX_FILENAME,
                revision="aca45de6945b5dc6399abcd2a9c55ded5dc9111f",
                max_length=8192,
            )
            reranked_retriever = RerankingMemoryRetriever(
                reranked_hybrid_retriever,
                reranker,
                candidate_k=20,
            )

            baseline_evaluator = MemoryRetrievalEvaluator(
                baseline_retriever,
                k=5,
            )
            reranked_evaluator = MemoryRetrievalEvaluator(
                reranked_retriever,
                k=5,
            )

            baseline = await baseline_evaluator.evaluate(MEMORY_RETRIEVAL_QUALITY_CASES)
            reranked = await reranked_evaluator.evaluate(MEMORY_RETRIEVAL_QUALITY_CASES)

            print()
            print("PostgreSQL hybrid@20 baseline:")
            print(f"recall@5    = {baseline.recall_at_k:.6f}")
            print(f"precision@5 = {baseline.precision_at_k:.6f}")
            print(f"mrr         = {baseline.mrr:.6f}")
            print(f"ndcg@5      = {baseline.ndcg_at_k:.6f}")
            print(f"latency_ms  = {baseline.mean_latency_ms:.3f}")

            print()
            print("PostgreSQL hybrid@20 + cross-encoder reranker:")
            print(f"recall@5    = {reranked.recall_at_k:.6f}")
            print(f"precision@5 = {reranked.precision_at_k:.6f}")
            print(f"mrr         = {reranked.mrr:.6f}")
            print(f"ndcg@5      = {reranked.ndcg_at_k:.6f}")
            print(f"latency_ms  = {reranked.mean_latency_ms:.3f}")

            print()
            print("Reranker delta:")
            print(f"recall@5    = " f"{reranked.recall_at_k - baseline.recall_at_k:+.6f}")
            print(f"precision@5 = " f"{reranked.precision_at_k - baseline.precision_at_k:+.6f}")
            print(f"mrr         = " f"{reranked.mrr - baseline.mrr:+.6f}")
            print(f"ndcg@5      = " f"{reranked.ndcg_at_k - baseline.ndcg_at_k:+.6f}")
            print(f"latency_ms  = " f"{reranked.mean_latency_ms - baseline.mean_latency_ms:+.3f}")

            assert baseline.evaluated_queries == len(MEMORY_RETRIEVAL_QUALITY_CASES)
            assert baseline.successful_queries == len(MEMORY_RETRIEVAL_QUALITY_CASES)
            assert baseline.failed_queries == 0

            assert reranked.evaluated_queries == len(MEMORY_RETRIEVAL_QUALITY_CASES)
            assert reranked.successful_queries == len(MEMORY_RETRIEVAL_QUALITY_CASES)
            assert reranked.failed_queries == 0

            assert baseline.mean_latency_ms >= 0.0
            assert reranked.mean_latency_ms >= 0.0
        finally:
            await _cleanup(
                [item.id for item in MEMORY_RETRIEVAL_QUALITY_ITEMS],
                session_factory,
            )
            engine.dispose()

    asyncio.run(run())
