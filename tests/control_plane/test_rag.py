from __future__ import annotations

from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from app.control_plane.app import app
from app.control_plane.dependencies import (
    get_rag_indexer,
    get_rag_query_service,
    get_rag_state_repository,
    get_rag_vector_store,
)
from rag import RAGIndexer
from rag.chunking import RecursiveChunker
from rag.models import Document, EmbeddingIdentity, EmbeddingResult
from rag.query import RAGQueryService
from rag.retrieval import SemanticRetriever
from rag.stores import InMemoryVectorStore


class FakeEmbeddingService:
    async def embed(self, text: str):
        result = await self.embed_with_metadata(text)
        return result.vector

    async def embed_with_metadata(self, text: str):
        checksum = sum(ord(character) for character in text)

        vector = (
            float(checksum),
            float(len(text)),
        )

        identity = EmbeddingIdentity(
            requested_provider="test-requested-provider",
            requested_model="test-logical-embedding",
            resolved_provider="test-resolved-provider",
            resolved_model="test-physical-embedding",
            dimension=len(vector),
        )

        return EmbeddingResult(
            vector=vector,
            identity=identity,
        )


class FakeProvenanceEmbeddingService:
    async def embed(self, text: str):
        result = await self.embed_with_metadata(text)
        return result.vector

    async def embed_with_metadata(self, text: str):
        checksum = sum(ord(character) for character in text)

        vector = (
            float(checksum),
            float(len(text)),
        )

        identity = EmbeddingIdentity(
            requested_provider="test-requested-provider",
            requested_model="test-logical-embedding",
            resolved_provider="test-resolved-provider",
            resolved_model="test-physical-embedding",
            dimension=len(vector),
        )

        return EmbeddingResult(
            vector=vector,
            identity=identity,
        )


class FakeRAGStateRepository:
    def __init__(self) -> None:
        self.saved = []

    def get_chunks(self, document_id: str):
        for record in reversed(self.saved):
            if record["document"].id == document_id:
                return record["chunks"]
        return []

    def get_document(self, document_id: str):
        for record in reversed(self.saved):
            if record["document"].id == document_id:
                return record["document"]
        return None

    def save_document(
        self,
        document,
        chunks,
        *,
        embedding_model=None,
        embedding_dimension=None,
    ) -> None:
        self.saved.append(
            {
                "document": document,
                "chunks": list(chunks),
                "embedding_model": embedding_model,
                "embedding_dimension": embedding_dimension,
            }
        )

    def delete_document(self, document_id: str) -> None:
        self.saved = [record for record in self.saved if record["document"].id != document_id]

    def save_indexed_document(
        self,
        document,
        embedded_chunks,
    ) -> None:
        self.saved.append(
            {
                "document": document,
                "chunks": [embedded_chunk.chunk for embedded_chunk in embedded_chunks],
                "embedded_chunks": list(embedded_chunks),
                "embedding_model": (
                    embedded_chunks[0].embedding_identity.resolved_model
                    if embedded_chunks and embedded_chunks[0].embedding_identity is not None
                    else None
                ),
                "embedding_dimension": (
                    embedded_chunks[0].embedding_identity.dimension
                    if embedded_chunks and embedded_chunks[0].embedding_identity is not None
                    else (len(embedded_chunks[0].embedding) if embedded_chunks else None)
                ),
            }
        )

    def ensure_document(self, document) -> None:
        return None


class FakeRAGVectorStore:
    def __init__(self) -> None:
        self.delete_chunks = AsyncMock()


class FakeChatService:
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        user_id: str | None = None,
    ):
        return {
            "reply": "The enterprise platform supports RAG and model routing.",
        }


def build_test_indexer() -> RAGIndexer:
    return RAGIndexer(
        chunker=RecursiveChunker(
            chunk_size=100,
            overlap=10,
        ),
        embedding_service=FakeProvenanceEmbeddingService(),
        vector_store=InMemoryVectorStore(),
    )


def build_test_query_service() -> RAGQueryService:
    vector_store = InMemoryVectorStore()
    embedding_service = FakeEmbeddingService()

    indexer = RAGIndexer(
        chunker=RecursiveChunker(
            chunk_size=100,
            overlap=10,
        ),
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    retriever = __import__(
        "rag.retrieval.retriever",
        fromlist=["SemanticRetriever"],
    ).SemanticRetriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    chat_service = FakeChatService()

    service = RAGQueryService(
        retriever=retriever,
        chat_service=chat_service,
    )

    service._test_indexer = indexer

    return service


client = TestClient(app)
error_client = TestClient(app, raise_server_exceptions=False)

AUTH_HEADERS = {"x-api-key": "super-secret-key"}


def teardown_function() -> None:
    app.dependency_overrides.clear()


def test_control_plane_rag_index_executes() -> None:
    indexer = build_test_indexer()
    state_repository = FakeRAGStateRepository()

    app.dependency_overrides[get_rag_indexer] = lambda: indexer
    app.dependency_overrides[get_rag_state_repository] = lambda: state_repository

    response = client.post(
        "/api/v1/rag/index",
        json={
            "document_id": "doc-control-plane",
            "content": "Enterprise AI Platform supports retrieval augmented generation.",
            "metadata": {
                "source": "architecture.md",
            },
        },
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["document_id"] == "doc-control-plane"
    assert payload["chunks_indexed"] == 1

    assert len(state_repository.saved) == 1

    saved = state_repository.saved[0]

    assert saved["document"].id == "doc-control-plane"
    assert saved["document"].metadata["source"] == "architecture.md"
    assert len(saved["chunks"]) == 1
    assert saved["chunks"][0].id == "doc-control-plane:chunk:0"
    assert saved["embedding_model"] == "test-physical-embedding"
    assert saved["embedding_dimension"] == 2

    embedded_chunk = saved["embedded_chunks"][0]
    identity = embedded_chunk.embedding_identity

    assert identity is not None
    assert identity.requested_provider == "test-requested-provider"
    assert identity.requested_model == "test-logical-embedding"
    assert identity.resolved_provider == "test-resolved-provider"
    assert identity.resolved_model == "test-physical-embedding"
    assert identity.dimension == 2


def test_control_plane_rag_delete_removes_vectors_and_state() -> None:
    indexer = build_test_indexer()
    state_repository = FakeRAGStateRepository()
    vector_store = FakeRAGVectorStore()

    document = Document(
        id="doc-delete",
        content="Enterprise AI Platform deletion test.",
        metadata={"source": "test"},
    )

    import asyncio

    async def seed_document() -> None:
        embedded_chunks = await indexer.index(document)
        state_repository.save_document(
            document,
            [embedded_chunk.chunk for embedded_chunk in embedded_chunks],
            embedding_model="test-embedding",
            embedding_dimension=2,
        )

    asyncio.run(seed_document())

    assert state_repository.get_chunks("doc-delete")

    app.dependency_overrides[get_rag_vector_store] = lambda: vector_store
    app.dependency_overrides[get_rag_state_repository] = lambda: state_repository

    response = client.delete(
        "/api/v1/rag/doc-delete",
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 204
    assert response.content == b""

    vector_store.delete_chunks.assert_awaited_once_with(
        ["doc-delete:chunk:0"],
    )

    assert state_repository.get_chunks("doc-delete") == []


def test_control_plane_rag_delete_missing_document_is_idempotent() -> None:
    state_repository = FakeRAGStateRepository()
    vector_store = FakeRAGVectorStore()

    app.dependency_overrides[get_rag_vector_store] = lambda: vector_store
    app.dependency_overrides[get_rag_state_repository] = lambda: state_repository

    response = client.delete(
        "/api/v1/rag/missing-document",
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 204
    vector_store.delete_chunks.assert_not_awaited()


def test_control_plane_rag_delete_does_not_remove_state_when_vector_delete_fails() -> None:
    state_repository = FakeRAGStateRepository()
    vector_store = FakeRAGVectorStore()

    document = Document(
        id="doc-vector-failure",
        content="Vector deletion failure test.",
        metadata={},
    )

    chunk = __import__(
        "rag.models",
        fromlist=["DocumentChunk"],
    ).DocumentChunk(
        id="doc-vector-failure:chunk:0",
        document_id="doc-vector-failure",
        content="Vector deletion failure test.",
        chunk_index=0,
    )

    state_repository.save_document(
        document,
        [chunk],
    )

    vector_store.delete_chunks.side_effect = RuntimeError(
        "vector deletion failed",
    )

    app.dependency_overrides[get_rag_vector_store] = lambda: vector_store
    app.dependency_overrides[get_rag_state_repository] = lambda: state_repository

    response = error_client.delete(
        "/api/v1/rag/doc-vector-failure",
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 500
    assert state_repository.get_document("doc-vector-failure") == document
    assert state_repository.get_chunks("doc-vector-failure") == [chunk]


def test_control_plane_rag_index_rejects_empty_content() -> None:
    indexer = build_test_indexer()

    app.dependency_overrides[get_rag_indexer] = lambda: indexer

    response = client.post(
        "/api/v1/rag/index",
        json={
            "document_id": "doc-empty",
            "content": "",
        },
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 422


def test_control_plane_rag_query_executes() -> None:
    vector_store = InMemoryVectorStore()
    embedding_service = FakeEmbeddingService()

    indexer = RAGIndexer(
        chunker=RecursiveChunker(
            chunk_size=100,
            overlap=10,
        ),
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    retriever = __import__(
        "rag.retrieval.retriever",
        fromlist=["SemanticRetriever"],
    ).SemanticRetriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    service = RAGQueryService(
        retriever=retriever,
        chat_service=FakeChatService(),
    )

    async def seed_document() -> None:
        await indexer.index(
            __import__("rag.models", fromlist=["Document"]).Document(
                id="doc-query",
                content="Enterprise AI Platform supports RAG and model routing.",
                metadata={
                    "source": "architecture.md",
                },
            )
        )

    import asyncio

    asyncio.run(seed_document())

    app.dependency_overrides[get_rag_query_service] = lambda: service

    response = client.post(
        "/api/v1/rag/query",
        json={
            "query": "What does the platform support?",
            "top_k": 1,
        },
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["answer"] == "The enterprise platform supports RAG and model routing."
    assert payload["retrieved_count"] == 1
    assert len(payload["sources"]) == 1

    source = payload["sources"][0]

    assert source["document_id"] == "doc-query"
    assert source["chunk_id"] == "doc-query:chunk:0"
    assert source["content"] == ("Enterprise AI Platform supports RAG and model routing.")
    assert source["metadata"]["source"] == "architecture.md"
    assert 0.0 <= source["score"] <= 1.0


def test_control_plane_rag_query_rejects_invalid_top_k() -> None:
    service = build_test_query_service()

    app.dependency_overrides[get_rag_query_service] = lambda: service

    response = client.post(
        "/api/v1/rag/query",
        json={
            "query": "What does the platform support?",
            "top_k": 0,
        },
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 422


def test_control_plane_rag_query_rejects_empty_query() -> None:
    service = build_test_query_service()

    app.dependency_overrides[get_rag_query_service] = lambda: service

    response = client.post(
        "/api/v1/rag/query",
        json={
            "query": "",
            "top_k": 5,
        },
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 422


def test_control_plane_rag_query_passes_retrieval_controls() -> None:
    class RecordingService:
        def __init__(self) -> None:
            self.calls = []

        async def query(
            self,
            *,
            query,
            top_k=5,
            min_score=None,
            metadata_filter=None,
            temperature=0.2,
            max_tokens=1024,
            user_id=None,
        ):
            self.calls.append(
                {
                    "query": query,
                    "top_k": top_k,
                    "min_score": min_score,
                    "metadata_filter": metadata_filter,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "user_id": user_id,
                }
            )

            return type(
                "Result",
                (),
                {
                    "answer": "test answer",
                    "sources": [],
                    "retrieved_count": 0,
                },
            )()

    service = RecordingService()

    app.dependency_overrides[get_rag_query_service] = lambda: service

    metadata_filter = {
        "tenant_id": "tenant-a",
        "source": "architecture.md",
    }

    response = client.post(
        "/api/v1/rag/query",
        json={
            "query": "enterprise architecture",
            "top_k": 5,
            "min_score": 0.75,
            "metadata_filter": metadata_filter,
        },
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200
    assert service.calls == [
        {
            "query": "enterprise architecture",
            "top_k": 5,
            "min_score": 0.75,
            "metadata_filter": metadata_filter,
            "temperature": 0.2,
            "max_tokens": 1024,
            "user_id": None,
        }
    ]


def test_control_plane_rag_query_rejects_invalid_min_score() -> None:
    service = build_test_query_service()

    app.dependency_overrides[get_rag_query_service] = lambda: service

    response = client.post(
        "/api/v1/rag/query",
        json={
            "query": "RAG",
            "min_score": 1.1,
        },
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 422


def test_control_plane_rag_query_rejects_invalid_metadata_filter() -> None:
    service = build_test_query_service()

    app.dependency_overrides[get_rag_query_service] = lambda: service

    response = client.post(
        "/api/v1/rag/query",
        json={
            "query": "RAG",
            "metadata_filter": ["tenant-a"],
        },
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 422


def test_control_plane_rag_query_applies_metadata_filter_end_to_end() -> None:
    vector_store = InMemoryVectorStore()
    embedding_service = FakeEmbeddingService()

    indexer = RAGIndexer(
        chunker=RecursiveChunker(
            chunk_size=100,
            overlap=10,
        ),
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    retriever = SemanticRetriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    service = RAGQueryService(
        retriever=retriever,
        chat_service=FakeChatService(),
    )

    import asyncio

    async def seed_documents() -> None:
        await indexer.index(
            Document(
                id="doc-architecture",
                content="The architecture uses Kafka, Spark, and Delta Lake.",
                metadata={
                    "tenant_id": "tenant-a",
                    "document_type": "architecture",
                },
            )
        )

        await indexer.index(
            Document(
                id="doc-policy",
                content="The enterprise policy requires approved model usage.",
                metadata={
                    "tenant_id": "tenant-b",
                    "document_type": "policy",
                },
            )
        )

    asyncio.run(seed_documents())

    app.dependency_overrides[get_rag_query_service] = lambda: service

    response = client.post(
        "/api/v1/rag/query",
        json={
            "query": "enterprise platform architecture",
            "top_k": 5,
            "metadata_filter": {
                "tenant_id": "tenant-a",
            },
        },
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["retrieved_count"] == 1
    assert len(payload["sources"]) == 1

    source = payload["sources"][0]

    assert source["document_id"] == "doc-architecture"
    assert source["metadata"] == {
        "tenant_id": "tenant-a",
        "document_type": "architecture",
    }
