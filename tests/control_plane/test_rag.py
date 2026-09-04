from __future__ import annotations

from fastapi.testclient import TestClient

from app.control_plane.app import app
from app.control_plane.dependencies import (
    get_rag_indexer,
    get_rag_query_service,
)
from rag import RAGIndexer
from rag.chunking import RecursiveChunker
from rag.query import RAGQueryService
from rag.stores import InMemoryVectorStore


class FakeEmbeddingService:
    async def embed(self, text: str):
        checksum = sum(ord(character) for character in text)

        return [
            float(checksum),
            float(len(text)),
        ]


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
        embedding_service=FakeEmbeddingService(),
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


def teardown_function() -> None:
    app.dependency_overrides.clear()


def test_control_plane_rag_index_executes() -> None:
    indexer = build_test_indexer()

    app.dependency_overrides[get_rag_indexer] = lambda: indexer

    response = client.post(
        "/api/v1/rag/index",
        json={
            "document_id": "doc-control-plane",
            "content": "Enterprise AI Platform supports retrieval augmented generation.",
            "metadata": {
                "source": "architecture.md",
            },
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["document_id"] == "doc-control-plane"
    assert payload["chunks_indexed"] == 1


def test_control_plane_rag_index_rejects_empty_content() -> None:
    indexer = build_test_indexer()

    app.dependency_overrides[get_rag_indexer] = lambda: indexer

    response = client.post(
        "/api/v1/rag/index",
        json={
            "document_id": "doc-empty",
            "content": "",
        },
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
    )

    assert response.status_code == 422
