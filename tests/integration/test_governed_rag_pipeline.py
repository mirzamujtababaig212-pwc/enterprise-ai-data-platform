from __future__ import annotations

import pytest

from rag.chunking.recursive import RecursiveChunker
from rag.governance import GovernancePolicy
from rag.indexing import RAGIndexer
from rag.models import Document
from rag.retrieval import SemanticRetriever
from rag.stores import InMemoryVectorStore


class ConstantEmbeddingService:
    async def embed(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("Text must not be empty.")
        return [1.0, 0.0]


@pytest.mark.asyncio
async def test_governance_policy_filters_rag_results() -> None:
    embedding_service = ConstantEmbeddingService()
    vector_store = InMemoryVectorStore()

    indexer = RAGIndexer(
        chunker=RecursiveChunker(
            chunk_size=500,
            overlap=0,
        ),
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    allowed_document = Document(
        id="doc-internal",
        content="Internal enterprise AI architecture guidance.",
        metadata={
            "classification": "internal",
            "allowed_use": "enterprise_ai",
        },
    )

    restricted_document = Document(
        id="doc-restricted",
        content="Restricted enterprise security architecture guidance.",
        metadata={
            "classification": "restricted",
            "allowed_use": "enterprise_ai",
        },
    )

    allowed_chunks = await indexer.index(allowed_document)
    restricted_chunks = await indexer.index(restricted_document)

    assert len(allowed_chunks) == 1
    assert len(restricted_chunks) == 1

    assert allowed_chunks[0].chunk.metadata == {
        "classification": "internal",
        "allowed_use": "enterprise_ai",
    }

    assert restricted_chunks[0].chunk.metadata == {
        "classification": "restricted",
        "allowed_use": "enterprise_ai",
    }

    retriever = SemanticRetriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    policy = GovernancePolicy(
        required_metadata={
            "classification": "internal",
            "allowed_use": "enterprise_ai",
        }
    )

    results = await retriever.retrieve(
        "enterprise architecture guidance",
        top_k=5,
        governance_policy=policy,
    )

    assert len(results) == 1
    assert results[0].chunk.document_id == "doc-internal"
    assert results[0].chunk.metadata["classification"] == "internal"
    assert results[0].chunk.metadata["allowed_use"] == "enterprise_ai"
