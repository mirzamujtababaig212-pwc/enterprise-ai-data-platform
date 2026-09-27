from __future__ import annotations

import os
from datetime import UTC, datetime

import pytest
from qdrant_client import AsyncQdrantClient
from sqlalchemy import create_engine, delete, text
from sqlalchemy.orm import Session, sessionmaker

from app.control_plane.persistence.models import RAGChunkRecord, RAGDocumentRecord
from rag.evaluation.dataset import RetrievalEvaluationDataset
from rag.evaluation.evaluator import RetrievalEvaluator
from rag.evaluation.lineage import (
    HybridRetrievalConfiguration,
    RetrievalEvaluationArtifact,
)
from rag.evaluation.models import RetrievalEvaluationCase
from rag.evaluation.policy import RetrievalEvaluationPolicy
from rag.evaluation.comparison.regression_policy import RetrievalRegressionPolicy
from rag.evaluation.models import RetrievalEvaluationResult
from rag.evaluation.stores.postgres import PostgreSQLRetrievalEvaluationRunStore
from rag.evaluation.workflow import RetrievalEvaluationWorkflow
from rag.evaluation.datasets.vehicle_retrieval_quality import (
    VEHICLE_QUALITY_EMBEDDING_IDENTITY,
    VehicleQualityBenchmarkEmbeddingService,
    vehicle_quality_benchmark_chunks,
)
from rag.models import DocumentChunk, EmbeddedChunk
from rag.retrieval import (
    HybridRetriever,
    PostgreSQLLexicalRetriever,
    SemanticRetriever,
)
from rag.stores.qdrant import QdrantVectorStore

pytestmark = pytest.mark.asyncio


def _postgres_engine():
    host = os.getenv("POSTGRES_TEST_HOST", "localhost")
    port = os.getenv("POSTGRES_TEST_PORT", "5432")
    user = os.getenv("POSTGRES_TEST_USER", "postgres")
    password = os.getenv("POSTGRES_TEST_PASSWORD", "postgres")
    database = os.getenv("POSTGRES_TEST_DB", "vehicle_platform")

    return create_engine(
        f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}",
        pool_pre_ping=True,
        future=True,
    )


def _seed_postgres(engine, document_id: str) -> None:
    chunks = vehicle_quality_benchmark_chunks()

    session_factory = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )
    session: Session = session_factory()

    try:
        session.add(
            RAGDocumentRecord(
                document_id=document_id,
                content="Cross-backend hybrid retrieval integration test",
                document_metadata={"test": True},
            )
        )

        session.add_all(
            [
                RAGChunkRecord(
                    chunk_id=f"{document_id}:{item.chunk.id}",
                    document_id=document_id,
                    chunk_index=index,
                    content=item.chunk.content,
                    chunk_metadata=dict(item.chunk.metadata),
                )
                for index, item in enumerate(chunks)
            ]
        )

        session.commit()
    finally:
        session.close()


def _cleanup_postgres(engine, document_id: str) -> None:
    session_factory = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )
    session: Session = session_factory()

    try:
        session.execute(
            delete(RAGChunkRecord).where(
                RAGChunkRecord.document_id == document_id,
            )
        )
        session.execute(
            delete(RAGDocumentRecord).where(
                RAGDocumentRecord.document_id == document_id,
            )
        )
        session.commit()
    finally:
        session.close()


async def test_real_hybrid_retrieval_quality_contract() -> None:
    if os.getenv("RUN_POSTGRES_INTEGRATION") != "1":
        pytest.skip("Set RUN_POSTGRES_INTEGRATION=1 to run PostgreSQL integration")

    if os.getenv("RUN_QDRANT_INTEGRATION") != "1":
        pytest.skip("Set RUN_QDRANT_INTEGRATION=1 to run Qdrant integration")

    postgres = _postgres_engine()

    qdrant_url = os.getenv("QDRANT_TEST_URL", "http://localhost:6333")
    collection = "hybrid-cross-backend-quality-contract"
    document_id = "hybrid-cross-backend-quality-contract"

    client = AsyncQdrantClient(url=qdrant_url)

    try:
        with postgres.connect() as connection:
            connection.execute(text("SELECT 1"))

        _seed_postgres(postgres, document_id)

        chunks = tuple(
            EmbeddedChunk(
                chunk=DocumentChunk(
                    id=f"{document_id}:{item.chunk.id}",
                    document_id=document_id,
                    content=item.chunk.content,
                    metadata=dict(item.chunk.metadata),
                    chunk_index=item.chunk.chunk_index,
                ),
                embedding=item.embedding,
                embedding_identity=VEHICLE_QUALITY_EMBEDDING_IDENTITY,
            )
            for item in vehicle_quality_benchmark_chunks()
        )

        if await client.collection_exists(collection):
            await client.delete_collection(collection)

        qdrant_store = QdrantVectorStore(
            client=client,
            collection_name=collection,
        )
        await qdrant_store.upsert(chunks)

        semantic_retriever = SemanticRetriever(
            embedding_service=VehicleQualityBenchmarkEmbeddingService(),
            vector_store=qdrant_store,
        )

        lexical_retriever = PostgreSQLLexicalRetriever(
            session_factory=sessionmaker(
                bind=postgres,
                autoflush=False,
                autocommit=False,
                expire_on_commit=False,
            ),
        )

        hybrid = HybridRetriever(
            semantic_retriever=semantic_retriever,
            lexical_retriever=lexical_retriever,
            candidate_k=5,
            rrf_k=60,
            semantic_weight=1.0,
            lexical_weight=0.5,
        )

        from rag.evaluation.datasets.vehicle_retrieval_quality import (
            vehicle_quality_evaluation_cases,
        )

        dataset = RetrievalEvaluationDataset.from_cases(
            "vehicle-retrieval-quality-real-hybrid",
            tuple(
                RetrievalEvaluationCase(
                    query=case.query,
                    relevant_chunk_ids=tuple(
                        f"{document_id}:{chunk_id}" for chunk_id in case.relevant_chunk_ids
                    ),
                    relevance_grades=(
                        {
                            f"{document_id}:{chunk_id}": grade
                            for chunk_id, grade in case.relevance_grades.items()
                        }
                        if case.relevance_grades is not None
                        else None
                    ),
                    expect_abstention=case.expect_abstention,
                )
                for case in vehicle_quality_evaluation_cases()
            ),
            version="v1",
        )

        evaluator = RetrievalEvaluator(
            hybrid,
            k=5,
            embedding_identity=VEHICLE_QUALITY_EMBEDDING_IDENTITY,
        )

        policy = RetrievalEvaluationPolicy(
            name="vehicle-retrieval-quality-real-hybrid-v1",
            min_recall_at_k=1.0,
            min_precision_at_k=0.2875,
            min_mrr=0.92,
            min_ndcg_at_k=0.93,
        )

        artifact = RetrievalEvaluationArtifact(
            retriever_type="HybridRetriever",
            vector_store_type="QdrantVectorStore+PostgreSQLLexicalRetriever",
            hybrid_configuration=HybridRetrievalConfiguration(
                candidate_k=5,
                rrf_k=60,
                semantic_weight=1.0,
                lexical_weight=0.5,
            ),
        )

        workflow = RetrievalEvaluationWorkflow(
            evaluator=evaluator,
            policy=policy,
            retrieval_artifact=artifact,
        )

        result = await workflow.run(dataset)

        assert result.passed is True
        assert result.evaluation.evaluated_queries == 16
        assert result.evaluation.successful_queries == 16
        assert result.evaluation.failed_queries == 0

        assert result.evaluation.recall_at_k == pytest.approx(1.0)
        assert result.evaluation.precision_at_k >= 0.2875
        assert result.evaluation.mrr >= 0.92
        assert result.evaluation.ndcg_at_k >= 0.93

        assert result.lineage.retrieval_artifact is not None
        assert result.lineage.retrieval_artifact.retriever_type == "HybridRetriever"
        assert (
            result.lineage.retrieval_artifact.vector_store_type
            == "QdrantVectorStore+PostgreSQLLexicalRetriever"
        )

        configuration = result.lineage.retrieval_artifact.hybrid_configuration
        assert configuration is not None
        assert configuration.candidate_k == 5
        assert configuration.rrf_k == 60
        assert configuration.semantic_weight == pytest.approx(1.0)
        assert configuration.lexical_weight == pytest.approx(0.5)

        lineage = result.as_dict()["lineage"]

        assert lineage["retriever_type"] == "HybridRetriever"
        assert lineage["vector_store_type"] == "QdrantVectorStore+PostgreSQLLexicalRetriever"
        assert lineage["hybrid_configuration"] == {
            "candidate_k": 5,
            "rrf_k": 60,
            "semantic_weight": 1.0,
            "lexical_weight": 0.5,
        }

    finally:
        _cleanup_postgres(postgres, document_id)
        postgres.dispose()

        if await client.collection_exists(collection):
            await client.delete_collection(collection)

        await client.close()


async def test_real_hybrid_baseline_persists_and_drives_regression_decision() -> None:
    if os.getenv("RUN_POSTGRES_INTEGRATION") != "1":
        pytest.skip("Set RUN_POSTGRES_INTEGRATION=1 to run PostgreSQL integration")

    if os.getenv("RUN_QDRANT_INTEGRATION") != "1":
        pytest.skip("Set RUN_QDRANT_INTEGRATION=1 to run Qdrant integration")

    postgres = _postgres_engine()

    baseline_run_id = "vehicle-real-hybrid-regression-baseline"
    candidate_run_id = "vehicle-real-hybrid-regression-candidate"

    qdrant_url = os.getenv("QDRANT_TEST_URL", "http://localhost:6333")
    collection = "hybrid-cross-backend-regression"
    document_id = "hybrid-cross-backend-regression"

    client = AsyncQdrantClient(url=qdrant_url)

    session_factory = sessionmaker(
        bind=postgres,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )

    try:
        with postgres.connect() as connection:
            connection.execute(text("SELECT 1"))

        _seed_postgres(postgres, document_id)

        chunks = tuple(
            EmbeddedChunk(
                chunk=DocumentChunk(
                    id=f"{document_id}:{item.chunk.id}",
                    document_id=document_id,
                    content=item.chunk.content,
                    metadata=dict(item.chunk.metadata),
                    chunk_index=item.chunk.chunk_index,
                ),
                embedding=item.embedding,
                embedding_identity=VEHICLE_QUALITY_EMBEDDING_IDENTITY,
            )
            for item in vehicle_quality_benchmark_chunks()
        )

        if await client.collection_exists(collection):
            await client.delete_collection(collection)

        qdrant_store = QdrantVectorStore(
            client=client,
            collection_name=collection,
        )
        await qdrant_store.upsert(chunks)

        semantic_retriever = SemanticRetriever(
            embedding_service=VehicleQualityBenchmarkEmbeddingService(),
            vector_store=qdrant_store,
        )

        lexical_retriever = PostgreSQLLexicalRetriever(
            session_factory=session_factory,
        )

        hybrid = HybridRetriever(
            semantic_retriever=semantic_retriever,
            lexical_retriever=lexical_retriever,
            candidate_k=5,
            rrf_k=60,
            semantic_weight=1.0,
            lexical_weight=0.5,
        )

        from rag.evaluation.datasets.vehicle_retrieval_quality import (
            vehicle_quality_evaluation_cases,
        )

        dataset = RetrievalEvaluationDataset.from_cases(
            "vehicle-retrieval-quality-real-hybrid",
            tuple(
                RetrievalEvaluationCase(
                    query=case.query,
                    relevant_chunk_ids=tuple(
                        f"{document_id}:{chunk_id}" for chunk_id in case.relevant_chunk_ids
                    ),
                    relevance_grades=(
                        {
                            f"{document_id}:{chunk_id}": grade
                            for chunk_id, grade in case.relevance_grades.items()
                        }
                        if case.relevance_grades is not None
                        else None
                    ),
                    expect_abstention=case.expect_abstention,
                )
                for case in vehicle_quality_evaluation_cases()
            ),
            version="v1",
        )

        evaluator = RetrievalEvaluator(
            hybrid,
            k=5,
            embedding_identity=VEHICLE_QUALITY_EMBEDDING_IDENTITY,
        )

        policy = RetrievalEvaluationPolicy(
            name="vehicle-retrieval-quality-real-hybrid-v1",
            min_recall_at_k=1.0,
            min_precision_at_k=0.2875,
            min_mrr=0.92,
            min_ndcg_at_k=0.93,
        )

        artifact = RetrievalEvaluationArtifact(
            retriever_type="HybridRetriever",
            vector_store_type="QdrantVectorStore+PostgreSQLLexicalRetriever",
            hybrid_configuration=HybridRetrievalConfiguration(
                candidate_k=5,
                rrf_k=60,
                semantic_weight=1.0,
                lexical_weight=0.5,
            ),
        )

        workflow = RetrievalEvaluationWorkflow(
            evaluator=evaluator,
            policy=policy,
            retrieval_artifact=artifact,
        )

        result = await workflow.run(dataset)

        assert result.passed is True
        assert result.evaluation.evaluated_queries == 16
        assert result.evaluation.successful_queries == 16
        assert result.evaluation.failed_queries == 0

        baseline = result.to_run(
            run_id=baseline_run_id,
            created_at=datetime(2026, 9, 27, 10, 0, tzinfo=UTC),
        )

        baseline_session = session_factory()
        try:
            repository = PostgreSQLRetrievalEvaluationRunStore(baseline_session)
            await repository.save(baseline)

            restored_baseline = await repository.get(baseline_run_id)
        finally:
            baseline_session.close()

        assert restored_baseline is not None
        assert restored_baseline.run_id == baseline_run_id
        assert restored_baseline.lineage == baseline.lineage
        assert restored_baseline.evaluation.recall_at_k == baseline.evaluation.recall_at_k
        assert restored_baseline.evaluation.precision_at_k == baseline.evaluation.precision_at_k
        assert restored_baseline.evaluation.mrr == baseline.evaluation.mrr
        assert restored_baseline.evaluation.ndcg_at_k == baseline.evaluation.ndcg_at_k
        assert restored_baseline.evaluation.evaluated_queries == 16
        assert restored_baseline.evaluation.successful_queries == 16
        assert restored_baseline.evaluation.failed_queries == 0
        assert restored_baseline.evaluation.query_results == ()
        assert restored_baseline.release_passed is True

        degraded_evaluation = RetrievalEvaluationResult(
            recall_at_k=0.99,
            precision_at_k=result.evaluation.precision_at_k,
            mrr=result.evaluation.mrr,
            ndcg_at_k=result.evaluation.ndcg_at_k,
            evaluated_queries=result.evaluation.evaluated_queries,
            successful_queries=result.evaluation.successful_queries,
            failed_queries=result.evaluation.failed_queries,
            mean_latency_ms=result.evaluation.mean_latency_ms,
            query_results=(),
            abstention_accuracy=result.evaluation.abstention_accuracy,
            abstention_evaluated_queries=result.evaluation.abstention_evaluated_queries,
        )

        degraded_result = result.__class__(
            dataset_name=result.dataset_name,
            evaluation=degraded_evaluation,
            quality_gate=result.quality_gate,
            lineage=result.lineage,
        )

        candidate = degraded_result.to_run(
            run_id=candidate_run_id,
            created_at=datetime(2026, 9, 27, 10, 1, tzinfo=UTC),
            baseline=restored_baseline,
            regression_policy=RetrievalRegressionPolicy(
                name="vehicle-real-hybrid-regression-v1",
                max_recall_at_k_degradation=0.0,
            ),
        )

        assert candidate.regression is not None
        assert candidate.regression.baseline_run_id == baseline_run_id
        assert candidate.regression.candidate_run_id == candidate_run_id
        assert candidate.regression.result.passed is False
        assert candidate.release_passed is False

        comparison = candidate.regression.comparison
        assert comparison.metrics["recall_at_k"].baseline == pytest.approx(
            baseline.evaluation.recall_at_k
        )
        assert comparison.metrics["recall_at_k"].candidate == pytest.approx(0.99)
        assert comparison.metrics["recall_at_k"].status.value == "regressed"

        candidate_session = session_factory()
        try:
            repository = PostgreSQLRetrievalEvaluationRunStore(candidate_session)
            await repository.save(candidate)

            restored_candidate = await repository.get(candidate_run_id)
        finally:
            candidate_session.close()

        assert restored_candidate is not None
        assert restored_candidate.lineage == candidate.lineage
        assert restored_candidate.regression is not None
        assert restored_candidate.regression.baseline_run_id == baseline_run_id
        assert restored_candidate.regression.candidate_run_id == candidate_run_id
        assert restored_candidate.regression.result.passed is False
        assert restored_candidate.release_passed is False
        assert (
            restored_candidate.regression.comparison.metrics["recall_at_k"].status.value
            == "regressed"
        )

    finally:
        cleanup_session = session_factory()
        try:
            cleanup_session.execute(
                text(
                    "DELETE FROM retrieval_evaluation_runs "
                    "WHERE run_id IN (:baseline_run_id, :candidate_run_id)"
                ),
                {
                    "baseline_run_id": baseline_run_id,
                    "candidate_run_id": candidate_run_id,
                },
            )
            cleanup_session.commit()
        finally:
            cleanup_session.close()

        _cleanup_postgres(postgres, document_id)
        postgres.dispose()

        if await client.collection_exists(collection):
            await client.delete_collection(collection)

        await client.close()


async def test_qdrant_semantic_postgres_lexical_hybrid_retrieval() -> None:
    if os.getenv("RUN_POSTGRES_INTEGRATION") != "1":
        pytest.skip("Set RUN_POSTGRES_INTEGRATION=1 to run PostgreSQL integration")

    if os.getenv("RUN_QDRANT_INTEGRATION") != "1":
        pytest.skip("Set RUN_QDRANT_INTEGRATION=1 to run Qdrant integration")

    postgres = _postgres_engine()

    qdrant_url = os.getenv("QDRANT_TEST_URL", "http://localhost:6333")
    collection = os.getenv(
        "QDRANT_TEST_COLLECTION",
        "hybrid-cross-backend-regression",
    )

    client = AsyncQdrantClient(url=qdrant_url)

    document_id = "hybrid-cross-backend-regression"

    try:
        with postgres.connect() as connection:
            connection.execute(text("SELECT 1"))

        _seed_postgres(postgres, document_id)

        chunks = tuple(
            EmbeddedChunk(
                chunk=DocumentChunk(
                    id=f"{document_id}:{item.chunk.id}",
                    document_id=document_id,
                    content=item.chunk.content,
                    metadata=dict(item.chunk.metadata),
                    chunk_index=item.chunk.chunk_index,
                ),
                embedding=item.embedding,
                embedding_identity=VEHICLE_QUALITY_EMBEDDING_IDENTITY,
            )
            for item in vehicle_quality_benchmark_chunks()
        )

        if await client.collection_exists(collection):
            await client.delete_collection(collection)

        qdrant_store = QdrantVectorStore(
            client=client,
            collection_name=collection,
        )
        await qdrant_store.upsert(chunks)

        semantic_retriever = SemanticRetriever(
            embedding_service=VehicleQualityBenchmarkEmbeddingService(),
            vector_store=qdrant_store,
        )

        lexical_retriever = PostgreSQLLexicalRetriever(
            session_factory=sessionmaker(
                bind=postgres,
                autoflush=False,
                autocommit=False,
                expire_on_commit=False,
            ),
        )

        hybrid = HybridRetriever(
            semantic_retriever=semantic_retriever,
            lexical_retriever=lexical_retriever,
            candidate_k=5,
            rrf_k=60,
            semantic_weight=1.0,
            lexical_weight=0.5,
        )

        query = "electric inverter battery motor power delivery"

        semantic_results = await semantic_retriever.retrieve(
            query,
            top_k=5,
        )
        lexical_results = await lexical_retriever.retrieve(
            query,
            top_k=5,
        )
        hybrid_results = await hybrid.retrieve(
            query,
            top_k=5,
        )

        assert semantic_results
        assert lexical_results
        assert hybrid_results

        semantic_ids = {result.chunk.id for result in semantic_results}
        lexical_ids = {result.chunk.id for result in lexical_results}
        hybrid_ids = [result.chunk.id for result in hybrid_results]

        candidate_ids = semantic_ids | lexical_ids
        shared_ids = semantic_ids & lexical_ids

        assert candidate_ids
        assert shared_ids
        assert set(hybrid_ids) <= candidate_ids
        assert len(hybrid_ids) == 5
        assert shared_ids & set(hybrid_ids)

        diagnostics = await hybrid.diagnose(
            query,
            top_k=5,
        )

        assert diagnostics
        assert diagnostics[0].final_rank == 1

        for item in diagnostics:
            assert item.fused_score == pytest.approx(
                item.semantic_contribution + item.lexical_contribution
            )

        assert any(
            item.semantic_rank is not None and item.lexical_rank is not None for item in diagnostics
        )

        assert all(
            result.embedding_identity == VEHICLE_QUALITY_EMBEDDING_IDENTITY
            for result in semantic_results
        )

    finally:
        _cleanup_postgres(postgres, document_id)
        postgres.dispose()

        if await client.collection_exists(collection):
            await client.delete_collection(collection)

        await client.close()
