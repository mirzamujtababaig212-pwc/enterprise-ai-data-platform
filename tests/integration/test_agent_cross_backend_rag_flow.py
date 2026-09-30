from __future__ import annotations

import os

import pytest
from qdrant_client import AsyncQdrantClient
from sqlalchemy import create_engine, delete, text
from sqlalchemy.orm import Session, sessionmaker

from ai_platform.agents.llm_agent import LLMAgent
from ai_platform.agents.evaluation.policy import AgentEvaluationPolicy
from ai_platform.agents.models import AgentDefinition, AgentRequest
from ai_platform.agents.registry.in_memory import InMemoryAgentRegistry
from ai_platform.agents.observability import AgentExecutionEventType
from ai_platform.agents.runtime import AgentRuntime
from ai_platform.llm_gateway.routing.router import Router
from app.control_plane.agent_run_events.postgres_observer import (
    PostgreSQLAgentRunEventObserver,
)
from app.control_plane.agent_evaluations.application_service import (
    AgentEvaluationApplicationService,
)
from app.control_plane.agent_evaluations.postgres_repository import (
    PostgreSQLAgentEvaluationRunsRepository,
)
from app.control_plane.agent_run_events.postgres_repository import (
    PostgreSQLAgentRunEventsRepository,
)
from app.control_plane.agent_runs.application_service import (
    AgentRunApplicationService,
)
from app.control_plane.agent_run_steps.postgres_repository import (
    PostgreSQLAgentRunStepsRepository,
)
from app.control_plane.agent_runs.postgres_repository import (
    PostgreSQLAgentRunRepository,
)
from app.control_plane.agent_runs.models import AgentRunStatus
from app.control_plane.persistence.models import (
    RAGChunkRecord,
    AgentRunRecord,
    RAGDocumentRecord,
)
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
from tools.rag.search import RAGSearchTool
from tools.registry.in_memory import InMemoryToolRegistry

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
                content="Agent cross-backend RAG integration test",
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
                    chunk_metadata={
                        **dict(item.chunk.metadata),
                        "tenant_id": "tenant-cross-backend-rag",
                    },
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


async def test_agent_run_executes_cross_backend_hybrid_rag() -> None:
    if os.getenv("RUN_POSTGRES_INTEGRATION") != "1":
        pytest.skip("Set RUN_POSTGRES_INTEGRATION=1 to run PostgreSQL integration")

    if os.getenv("RUN_QDRANT_INTEGRATION") != "1":
        pytest.skip("Set RUN_QDRANT_INTEGRATION=1 to run Qdrant integration")

    postgres = _postgres_engine()

    postgres_session_factory = sessionmaker(
        bind=postgres,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )

    qdrant_url = os.getenv(
        "QDRANT_TEST_URL",
        "http://localhost:6333",
    )
    collection = "agent-cross-backend-rag-regression"
    document_id = "agent-cross-backend-rag-regression"

    client = AsyncQdrantClient(url=qdrant_url)

    try:
        with postgres.connect() as connection:
            connection.execute(text("SELECT 1"))

        _seed_postgres(postgres, document_id)

        embedded_chunks = tuple(
            EmbeddedChunk(
                chunk=DocumentChunk(
                    id=f"{document_id}:{item.chunk.id}",
                    document_id=document_id,
                    content=item.chunk.content,
                    metadata={
                        **dict(item.chunk.metadata),
                        "tenant_id": "tenant-cross-backend-rag",
                    },
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
        await qdrant_store.upsert(embedded_chunks)

        embedding_service = VehicleQualityBenchmarkEmbeddingService()

        semantic_retriever = SemanticRetriever(
            embedding_service=embedding_service,
            vector_store=qdrant_store,
        )

        lexical_retriever = PostgreSQLLexicalRetriever(
            session_factory=postgres_session_factory,
        )

        hybrid_retriever = HybridRetriever(
            semantic_retriever=semantic_retriever,
            lexical_retriever=lexical_retriever,
            candidate_k=5,
            rrf_k=60,
            semantic_weight=1.0,
            lexical_weight=0.5,
        )

        query = "electric inverter battery motor power delivery"

        expected_results = await hybrid_retriever.retrieve(
            query,
            top_k=3,
        )

        assert expected_results

        tool_registry = InMemoryToolRegistry()
        await tool_registry.register(
            RAGSearchTool(hybrid_retriever),
        )

        agent_definition = AgentDefinition(
            name="enterprise-rag-analyst",
            description="Agent integration test for cross-backend hybrid RAG.",
            system_prompt=(
                "You are an enterprise RAG analyst. "
                "Use the rag.search tool when enterprise knowledge is required "
                "and ground your response in retrieved context."
            ),
            model="mock-gpt",
            temperature=0.2,
            max_tokens=1024,
            tool_names=("rag.search",),
        )

        event_observer = PostgreSQLAgentRunEventObserver(
            postgres_session_factory,
        )

        agent_registry = InMemoryAgentRegistry()
        await agent_registry.register(
            LLMAgent(
                agent_definition,
                observer=event_observer,
            ),
        )

        gateway = Router()

        step_repositories = []

        def agent_run_steps_repository_factory():
            repository = PostgreSQLAgentRunStepsRepository(
                postgres_session_factory(),
            )
            step_repositories.append(repository)
            return repository

        runtime = AgentRuntime(
            agent_registry,
            tool_registry=tool_registry,
            llm_gateway=gateway,
            agent_run_steps_repository_factory=agent_run_steps_repository_factory,
        )

        agent_run_session = postgres_session_factory()
        repository = PostgreSQLAgentRunRepository(agent_run_session)
        agent_run_steps_repository = PostgreSQLAgentRunStepsRepository(
            postgres_session_factory(),
        )

        application_service = AgentRunApplicationService(
            runtime=runtime,
            repository=repository,
            agent_run_steps_repository=agent_run_steps_repository,
        )

        result = await application_service.execute(
            agent_name="enterprise-rag-analyst",
            request=AgentRequest(
                input=query,
                session_id="session-cross-backend-rag-123",
                user_id="user-cross-backend-rag-456",
                principal="user-cross-backend-rag-456",
                tenant_id="tenant-cross-backend-rag",
                metadata={
                    "source": "agent-cross-backend-rag-integration",
                    "mock_tool_call": "rag.search",
                },
            ),
        )

        response = result.response

        assert result.run_id

        assert response.agent_name == "enterprise-rag-analyst"
        assert response.session_id == "session-cross-backend-rag-123"
        assert response.metadata["provider"] == "mock"
        assert response.metadata["model"] == "mock-gpt"
        assert response.metadata["tool_rounds"] == 1

        expected_content = expected_results[0].chunk.content

        assert expected_content in response.output

        persisted_run = repository.get(result.run_id)

        assert persisted_run is not None
        assert persisted_run.run_id == result.run_id
        assert persisted_run.agent_name == "enterprise-rag-analyst"
        assert persisted_run.session_id == "session-cross-backend-rag-123"
        assert persisted_run.user_id == "user-cross-backend-rag-456"
        assert persisted_run.principal == "user-cross-backend-rag-456"
        assert persisted_run.tenant_id == "tenant-cross-backend-rag"
        assert persisted_run.status == AgentRunStatus.COMPLETED
        assert persisted_run.output == response.output

        event_session = postgres_session_factory()
        try:
            event_repository = PostgreSQLAgentRunEventsRepository(
                event_session,
            )
            events = event_repository.list(result.run_id)
        finally:
            event_session.close()

        completed_tool_events = [
            event
            for event in events
            if (
                event.event_type == AgentExecutionEventType.TOOL_CALL_COMPLETED
                and event.tool_name == "rag.search"
            )
        ]

        assert len(completed_tool_events) == 1

        rag_completed_event = completed_tool_events[0]

        assert rag_completed_event.run_id == result.run_id
        assert rag_completed_event.session_id == "session-cross-backend-rag-123"
        assert rag_completed_event.call_id

        assert rag_completed_event.metadata["execution_provenance"] == {
            "execution_status": "completed",
            "tool_provider": {
                "kind": "native",
                "name": "internal",
            },
            "tenant_id": "tenant-cross-backend-rag",
        }

        assert rag_completed_event.metadata["rag_provenance"] == {
            "retrieved_count": len(expected_results),
            "sources": [
                {
                    "chunk_id": item.chunk.id,
                    "document_id": item.chunk.document_id,
                    "score": item.score,
                    "retrieval_score": item.score,
                    "reranker_score": item.reranker_score,
                }
                for item in expected_results
            ],
        }

        semantic_results = await semantic_retriever.retrieve(
            query,
            top_k=5,
        )
        lexical_results = await lexical_retriever.retrieve(
            query,
            top_k=5,
        )

        semantic_ids = {item.chunk.id for item in semantic_results}
        lexical_ids = {item.chunk.id for item in lexical_results}
        expected_ids = {item.chunk.id for item in expected_results}

        assert semantic_ids
        assert lexical_ids
        assert expected_ids
        assert expected_ids <= semantic_ids | lexical_ids

        assert all(
            item.embedding_identity == VEHICLE_QUALITY_EMBEDDING_IDENTITY
            for item in semantic_results
        )

        # Verify the RAG provenance survived from the runtime into the
        # durable agent-run step repository.
        persisted_steps = agent_run_steps_repository.list(
            result.run_id,
            limit=10_000,
        )

        rag_steps = [
            step
            for step in persisted_steps
            if step.tool_name == "rag.search" and step.metadata.get("rag_provenance")
        ]

        assert len(rag_steps) == 1

        rag_provenance = rag_steps[0].metadata["rag_provenance"]

        assert rag_provenance["retrieved_count"] == len(expected_results)
        assert rag_provenance["sources"] == [
            {
                "chunk_id": item.chunk.id,
                "document_id": item.chunk.document_id,
                "score": item.score,
                "retrieval_score": item.score,
                "reranker_score": item.reranker_score,
            }
            for item in expected_results
        ]

        # Evaluate the same durable run through the production evaluation
        # application service and persist the resulting evaluation artifact.
        evaluation_run_session = postgres_session_factory()
        evaluation_steps_session = postgres_session_factory()
        evaluation_events_session = postgres_session_factory()

        evaluation_repository = PostgreSQLAgentEvaluationRunsRepository(
            evaluation_run_session,
        )
        evaluation_steps_repository = PostgreSQLAgentRunStepsRepository(
            evaluation_steps_session,
        )
        evaluation_events_repository = PostgreSQLAgentRunEventsRepository(
            evaluation_events_session,
        )

        evaluation_service = AgentEvaluationApplicationService(
            agent_run_repository=repository,
            agent_run_steps_repository=evaluation_steps_repository,
            agent_run_events_repository=evaluation_events_repository,
            evaluation_repository=evaluation_repository,
        )

        evaluation = await evaluation_service.evaluate_run(
            result.run_id,
            tenant_id=persisted_run.tenant_id,
            principal=persisted_run.principal,
            policy=AgentEvaluationPolicy(
                max_execution_time_ms=60_000,
                max_steps_per_run=20,
                max_invalid_tool_calls=0,
                allow_governance_denials=False,
                require_task_completed=True,
                name="cross-backend-rag-regression",
            ),
        )

        retrieval_scores = [item.score for item in expected_results]
        assert retrieval_scores

        assert evaluation.metrics.retrieval_score_min == min(retrieval_scores)
        assert evaluation.metrics.retrieval_score_max == max(retrieval_scores)
        assert evaluation.metrics.retrieval_score_avg == pytest.approx(
            sum(retrieval_scores) / len(retrieval_scores),
        )

        reranker_scores = [
            item.reranker_score for item in expected_results if item.reranker_score is not None
        ]

        if reranker_scores:
            assert evaluation.metrics.reranker_score_min == min(reranker_scores)
            assert evaluation.metrics.reranker_score_max == max(reranker_scores)
            assert evaluation.metrics.reranker_score_avg == pytest.approx(
                sum(reranker_scores) / len(reranker_scores),
            )
        else:
            assert evaluation.metrics.reranker_score_min is None
            assert evaluation.metrics.reranker_score_max is None
            assert evaluation.metrics.reranker_score_avg is None

        # Verify the exact persisted JSON contains the diagnostics.
        evaluation_record = evaluation_repository.get(evaluation.evaluation_run_id)

        assert evaluation_record is not None
        assert evaluation_record.metrics.retrieval_score_min == (
            evaluation.metrics.retrieval_score_min
        )
        assert evaluation_record.metrics.retrieval_score_max == (
            evaluation.metrics.retrieval_score_max
        )
        assert evaluation_record.metrics.retrieval_score_avg == pytest.approx(
            evaluation.metrics.retrieval_score_avg,
        )

        assert evaluation_record.metrics.reranker_score_min == (
            evaluation.metrics.reranker_score_min
        )
        assert evaluation_record.metrics.reranker_score_max == (
            evaluation.metrics.reranker_score_max
        )
        assert evaluation_record.metrics.reranker_score_avg == (
            evaluation.metrics.reranker_score_avg
        )

        evaluation_events_session.close()
        evaluation_steps_session.close()
        evaluation_run_session.close()

    finally:
        for step_repository in locals().get("step_repositories", []):
            step_repository.close()

        if "agent_run_steps_repository" in locals():
            agent_run_steps_repository.close()

        if "evaluation_events_session" in locals():
            evaluation_events_session.close()

        if "evaluation_steps_session" in locals():
            evaluation_steps_session.close()

        if "evaluation_run_session" in locals():
            evaluation_run_session.close()

        if "evaluation_repository" in locals():
            evaluation_repository.close()

        if "agent_run_session" in locals():
            agent_run_session.close()

        if "result" in locals() and "postgres_session_factory" in locals():
            cleanup_session = postgres_session_factory()
            try:
                cleanup_session.execute(
                    delete(AgentRunRecord).where(
                        AgentRunRecord.run_id == result.run_id,
                    )
                )
                cleanup_session.commit()
            finally:
                cleanup_session.close()

        _cleanup_postgres(postgres, document_id)
        postgres.dispose()

        if await client.collection_exists(collection):
            await client.delete_collection(collection)

        await client.close()
