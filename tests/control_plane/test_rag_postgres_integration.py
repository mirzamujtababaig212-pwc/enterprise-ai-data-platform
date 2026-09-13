from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, delete, select, text
from sqlalchemy.orm import Session, sessionmaker

from app.control_plane.app import app
from app.control_plane.dependencies import (
    get_rag_indexer,
    get_rag_query_service,
    get_rag_state_repository,
)
from app.control_plane.persistence.models import (
    RAGChunkRecord,
    RAGDocumentRecord,
)
from app.control_plane.persistence.rag_state import PostgreSQLRAGStateRepository
from rag.chunking import RecursiveChunker
from rag.indexing import RAGIndexer
from rag.models import EmbeddingIdentity, EmbeddingResult
from rag.query import RAGQueryService
from rag.retrieval import SemanticRetriever
from rag.stores.postgres import PostgreSQLVectorStore


def _test_session_factory() -> tuple[sessionmaker[Session], object]:
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

    session_factory = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )

    return session_factory, engine


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_POSTGRES_INTEGRATION") != "1",
    reason="Set RUN_POSTGRES_INTEGRATION=1 to run the PostgreSQL integration test",
)


class DeterministicEmbeddingService:
    async def embed(self, text: str) -> tuple[float, float]:
        result = await self.embed_with_metadata(text)
        return result.vector

    async def embed_with_metadata(self, text: str) -> EmbeddingResult:
        return EmbeddingResult(
            vector=(1.0, 0.0),
            identity=EmbeddingIdentity(
                requested_provider="integration-test",
                requested_model="integration-test-embedding",
                resolved_provider="integration-test",
                resolved_model="integration-test-embedding",
                dimension=2,
            ),
        )


class FakeChatService:
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        user_id: str | None = None,
    ) -> dict[str, str]:
        return {
            "reply": "The indexed document was retrieved successfully.",
        }


def test_control_plane_rag_postgres_index_and_query_end_to_end() -> None:
    document_id = "postgres-control-plane-integration"
    client = TestClient(app)
    test_session_factory, engine = _test_session_factory()

    vector_store = PostgreSQLVectorStore(
        session_factory=test_session_factory,
    )
    embedding_service = DeterministicEmbeddingService()

    indexer = RAGIndexer(
        chunker=RecursiveChunker(
            chunk_size=1000,
            overlap=100,
        ),
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    retriever = SemanticRetriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    query_service = RAGQueryService(
        retriever=retriever,
        chat_service=FakeChatService(),
    )

    def state_repository_dependency():
        session = test_session_factory()
        try:
            yield PostgreSQLRAGStateRepository(session)
        finally:
            session.close()

    app.dependency_overrides[get_rag_indexer] = lambda: indexer
    app.dependency_overrides[get_rag_query_service] = lambda: query_service
    app.dependency_overrides[get_rag_state_repository] = state_repository_dependency

    try:
        index_response = client.post(
            "/api/v1/rag/index",
            json={
                "document_id": document_id,
                "content": (
                    "DELDAI Enterprise AI OS provides governed " "retrieval and agent capabilities."
                ),
                "metadata": {
                    "source": "postgres-integration-test",
                    "category": "architecture",
                },
            },
            headers={"x-api-key": "super-secret-key"},
        )

        assert index_response.status_code == 200
        assert index_response.json() == {
            "document_id": document_id,
            "chunks_indexed": 1,
        }

        with test_session_factory() as session:
            document = session.scalar(
                select(RAGDocumentRecord).where(RAGDocumentRecord.document_id == document_id)
            )
            assert document is not None

            chunks = list(
                session.scalars(
                    select(RAGChunkRecord)
                    .where(RAGChunkRecord.document_id == document_id)
                    .order_by(RAGChunkRecord.chunk_index.asc())
                ).all()
            )

            assert len(chunks) == 1

            chunk = chunks[0]
            assert chunk.embedding is not None
            assert len(chunk.embedding) == 2
            assert chunk.embedding_dimension == 2
            assert chunk.embedding_model == "integration-test-embedding"
            assert chunk.embedding_resolved_provider == "integration-test"
            assert chunk.embedding_resolved_model == "integration-test-embedding"

        query_response = client.post(
            "/api/v1/rag/query",
            json={
                "query": "What does DELDAI provide?",
                "top_k": 1,
            },
            headers={"x-api-key": "super-secret-key"},
        )

        assert query_response.status_code == 200

        payload = query_response.json()

        assert payload["answer"] == ("The indexed document was retrieved successfully.")
        assert payload["retrieved_count"] == 1
        assert len(payload["sources"]) == 1

        source = payload["sources"][0]

        assert source["document_id"] == document_id
        assert source["chunk_id"] == f"{document_id}:chunk:0"
        assert source["content"] == (
            "DELDAI Enterprise AI OS provides governed " "retrieval and agent capabilities."
        )
        assert source["metadata"] == {
            "source": "postgres-integration-test",
            "category": "architecture",
        }
        assert source["score"] == pytest.approx(1.0)

    finally:
        app.dependency_overrides.clear()

        with test_session_factory() as session:
            session.execute(delete(RAGChunkRecord).where(RAGChunkRecord.document_id == document_id))
            session.execute(
                delete(RAGDocumentRecord).where(RAGDocumentRecord.document_id == document_id)
            )
            session.commit()
        engine.dispose()
