from __future__ import annotations

import os

import pytest
from qdrant_client import AsyncQdrantClient
from sqlalchemy import create_engine, delete, text
from sqlalchemy.orm import Session, sessionmaker

from ai_platform.agents.llm_agent import LLMAgent
from ai_platform.agents.models import AgentDefinition, AgentRequest
from ai_platform.agents.registry.in_memory import InMemoryAgentRegistry
from ai_platform.agents.runtime import AgentRuntime
from ai_platform.llm_gateway.routing.router import Router
from app.control_plane.agent_runs.application_service import (
    AgentRunApplicationService,
)
from app.control_plane.agent_runs.in_memory import InMemoryAgentRunRepository
from app.control_plane.agent_runs.models import AgentRunStatus
from app.control_plane.persistence.models import (
    RAGChunkRecord,
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


async def test_agent_run_executes_cross_backend_hybrid_rag() -> None:
    if os.getenv("RUN_POSTGRES_INTEGRATION") != "1":
        pytest.skip("Set RUN_POSTGRES_INTEGRATION=1 to run PostgreSQL integration")

    if os.getenv("RUN_QDRANT_INTEGRATION") != "1":
        pytest.skip("Set RUN_QDRANT_INTEGRATION=1 to run Qdrant integration")

    postgres = _postgres_engine()

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
        await qdrant_store.upsert(embedded_chunks)

        embedding_service = VehicleQualityBenchmarkEmbeddingService()

        semantic_retriever = SemanticRetriever(
            embedding_service=embedding_service,
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
            name="enterprise-cross-backend-rag-test",
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

        agent_registry = InMemoryAgentRegistry()
        await agent_registry.register(
            LLMAgent(agent_definition),
        )

        gateway = Router()

        runtime = AgentRuntime(
            agent_registry,
            tool_registry=tool_registry,
            llm_gateway=gateway,
        )

        repository = InMemoryAgentRunRepository()

        application_service = AgentRunApplicationService(
            runtime=runtime,
            repository=repository,
        )

        result = await application_service.execute(
            agent_name="enterprise-cross-backend-rag-test",
            request=AgentRequest(
                input=query,
                session_id="session-cross-backend-rag-123",
                user_id="user-cross-backend-rag-456",
                metadata={
                    "source": "agent-cross-backend-rag-integration",
                    "mock_tool_call": "rag.search",
                },
            ),
        )

        response = result.response

        assert result.run_id

        assert response.agent_name == "enterprise-cross-backend-rag-test"
        assert response.session_id == "session-cross-backend-rag-123"
        assert response.metadata["provider"] == "mock"
        assert response.metadata["model"] == "mock-gpt"
        assert response.metadata["tool_rounds"] == 1

        expected_content = expected_results[0].chunk.content

        assert expected_content in response.output

        persisted_run = repository.get(result.run_id)

        assert persisted_run is not None
        assert persisted_run.run_id == result.run_id
        assert persisted_run.agent_name == "enterprise-cross-backend-rag-test"
        assert persisted_run.session_id == "session-cross-backend-rag-123"
        assert persisted_run.user_id == "user-cross-backend-rag-456"
        assert persisted_run.status == AgentRunStatus.COMPLETED
        assert persisted_run.output == response.output

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

    finally:
        _cleanup_postgres(postgres, document_id)
        postgres.dispose()

        if await client.collection_exists(collection):
            await client.delete_collection(collection)

        await client.close()
