from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from app.control_plane.app import app
from app.control_plane.routes.rag import _authenticated_tenant_id
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
            "reply": "The enterprise platform supports RAG and model routing. [Source 1]",
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
    assert saved["document"].metadata["tenant_id"] == "tenant-a"
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


def test_control_plane_rag_index_preserves_matching_tenant_metadata() -> None:
    indexer = build_test_indexer()
    state_repository = FakeRAGStateRepository()

    app.dependency_overrides[get_rag_indexer] = lambda: indexer
    app.dependency_overrides[get_rag_state_repository] = lambda: state_repository

    response = client.post(
        "/api/v1/rag/index",
        json={
            "document_id": "doc-control-plane-matching-tenant",
            "content": "Tenant-scoped enterprise document.",
            "metadata": {
                "source": "architecture.md",
                "tenant_id": "tenant-a",
            },
        },
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200
    saved = state_repository.saved[0]

    assert saved["document"].metadata["tenant_id"] == "tenant-a"


def test_control_plane_rag_index_rejects_conflicting_tenant_metadata() -> None:
    indexer = build_test_indexer()
    state_repository = FakeRAGStateRepository()

    app.dependency_overrides[get_rag_indexer] = lambda: indexer
    app.dependency_overrides[get_rag_state_repository] = lambda: state_repository

    response = client.post(
        "/api/v1/rag/index",
        json={
            "document_id": "doc-control-plane-conflicting-tenant",
            "content": "Conflicting tenant document.",
            "metadata": {
                "source": "architecture.md",
                "tenant_id": "tenant-b",
            },
        },
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Document metadata tenant_id conflicts with authenticated tenant."
    )
    assert state_repository.saved == []


def test_control_plane_rag_delete_removes_vectors_and_state() -> None:
    indexer = build_test_indexer()
    state_repository = FakeRAGStateRepository()
    vector_store = FakeRAGVectorStore()

    document = Document(
        id="doc-delete",
        content="Enterprise AI Platform deletion test.",
        metadata={"source": "test", "tenant_id": "tenant-a"},
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


def test_control_plane_rag_index_does_not_overwrite_another_tenant_document() -> None:
    indexer = build_test_indexer()
    state_repository = FakeRAGStateRepository()

    original = Document(
        id="doc-owned-by-tenant-b",
        content="Original tenant-b document.",
        metadata={"tenant_id": "tenant-b", "source": "original"},
    )
    state_repository.save_document(original, [])

    app.dependency_overrides[get_rag_indexer] = lambda: indexer
    app.dependency_overrides[get_rag_state_repository] = lambda: state_repository

    response = client.post(
        "/api/v1/rag/index",
        json={
            "document_id": original.id,
            "content": "Attempted tenant-a overwrite.",
            "metadata": {"source": "overwrite"},
        },
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 404
    assert state_repository.get_document(original.id) == original
    assert len(state_repository.saved) == 1


def test_control_plane_rag_delete_does_not_remove_another_tenant_document() -> None:
    state_repository = FakeRAGStateRepository()
    vector_store = FakeRAGVectorStore()

    document = Document(
        id="doc-owned-by-tenant-b",
        content="Private tenant-b document.",
        metadata={"tenant_id": "tenant-b"},
    )
    from rag.models import DocumentChunk

    chunk = DocumentChunk(
        id="doc-owned-by-tenant-b:chunk:0",
        document_id=document.id,
        content=document.content,
        chunk_index=0,
        metadata={"tenant_id": "tenant-b"},
    )
    state_repository.save_document(document, [chunk])

    app.dependency_overrides[get_rag_vector_store] = lambda: vector_store
    app.dependency_overrides[get_rag_state_repository] = lambda: state_repository

    response = client.delete(
        f"/api/v1/rag/{document.id}",
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 204
    assert state_repository.get_document(document.id) == document
    assert state_repository.get_chunks(document.id) == [chunk]
    vector_store.delete_chunks.assert_not_awaited()


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
        metadata={"tenant_id": "tenant-a"},
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
                    "tenant_id": "tenant-a",
                    "source": "architecture.md",
                    "source_ref": {
                        "platform": "snowflake",
                        "object_type": "table",
                        "object_name": "ANALYTICS.CUSTOMERS",
                        "namespace": "ANALYTICS",
                        "environment": "prod",
                    },
                    "locator": {
                        "type": "document_section",
                        "value": "customer_overview",
                    },
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

    assert payload["answer"] == "The enterprise platform supports RAG and model routing. [Source 1]"
    assert payload["retrieved_count"] == 1
    assert len(payload["sources"]) == 1

    source = payload["sources"][0]

    assert source["document_id"] == "doc-query"
    assert source["chunk_id"] == "doc-query:chunk:0"
    assert source["content"] == ("Enterprise AI Platform supports RAG and model routing.")
    assert source["metadata"]["source"] == "architecture.md"
    assert 0.0 <= source["score"] <= 1.0
    assert source["evidence_id"] == "evidence:doc-query:chunk:0"
    assert source["retrieval_rank"] == 1
    assert source["source_ref"] == {
        "platform": "snowflake",
        "object_type": "table",
        "object_name": "ANALYTICS.CUSTOMERS",
        "namespace": "ANALYTICS",
        "environment": "prod",
    }
    assert source["locator"] == {
        "type": "document_section",
        "value": "customer_overview",
    }
    assert source["metadata"]["source_ref"] == source["source_ref"]
    assert source["metadata"]["locator"] == source["locator"]


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
            governance_policy=None,
            tenant_id=None,
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
                    "governance_policy": governance_policy,
                    "tenant_id": tenant_id,
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
            "governance_policy": None,
            "tenant_id": "tenant-a",
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


def test_control_plane_rag_query_rejects_cross_tenant_metadata_filter() -> None:
    vector_store = InMemoryVectorStore()
    embedding_service = FakeEmbeddingService()

    retriever = SemanticRetriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    service = RAGQueryService(
        retriever=retriever,
        chat_service=FakeChatService(),
    )

    app.dependency_overrides[get_rag_query_service] = lambda: service

    response = client.post(
        "/api/v1/rag/query",
        json={
            "query": "enterprise policy",
            "top_k": 5,
            "metadata_filter": {
                "tenant_id": "tenant-b",
            },
        },
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 400
    assert "authenticated tenant" in response.json()["detail"]


def _request_with_tenant_state(state: dict) -> Request:
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/api/v1/rag/query",
        "raw_path": b"/api/v1/rag/query",
        "query_string": b"",
        "headers": [],
        "server": ("testserver", 80),
        "client": ("testclient", 50000),
        "state": state,
    }
    return Request(scope)


def test_authenticated_tenant_id_returns_tenant_from_request_state() -> None:
    request = _request_with_tenant_state({"tenant_id": "tenant-a"})

    assert _authenticated_tenant_id(request) == "tenant-a"


def test_authenticated_tenant_id_rejects_missing_tenant() -> None:
    request = _request_with_tenant_state({})

    with pytest.raises(HTTPException) as exc_info:
        _authenticated_tenant_id(request)

    assert exc_info.value.status_code == 403
    assert exc_info.value.detail == "Authenticated tenant identity is required."


def test_authenticated_tenant_id_rejects_empty_tenant() -> None:
    request = _request_with_tenant_state({"tenant_id": ""})

    with pytest.raises(HTTPException) as exc_info:
        _authenticated_tenant_id(request)

    assert exc_info.value.status_code == 403


def test_authenticated_tenant_id_rejects_whitespace_tenant() -> None:
    request = _request_with_tenant_state({"tenant_id": "   "})

    with pytest.raises(HTTPException) as exc_info:
        _authenticated_tenant_id(request)

    assert exc_info.value.status_code == 403


def test_authenticated_tenant_id_rejects_non_string_tenant() -> None:
    request = _request_with_tenant_state({"tenant_id": 123})

    with pytest.raises(HTTPException) as exc_info:
        _authenticated_tenant_id(request)

    assert exc_info.value.status_code == 403
