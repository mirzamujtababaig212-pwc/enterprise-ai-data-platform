from __future__ import annotations

import pytest

from rag.models import DocumentChunk, RetrievalResult
from tools.rag.search import RAGSearchTool


class FakeRetriever:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        self.calls.append((query, top_k))

        return [
            RetrievalResult(
                chunk=DocumentChunk(
                    id="chunk-1",
                    document_id="doc-1",
                    content="Enterprise RAG retrieves relevant knowledge.",
                    metadata={"source": "test"},
                    chunk_index=0,
                ),
                score=0.95,
            ),
            RetrievalResult(
                chunk=DocumentChunk(
                    id="chunk-2",
                    document_id="doc-1",
                    content="Retrieved context is supplied to the agent.",
                    metadata={"source": "test"},
                    chunk_index=1,
                ),
                score=0.85,
            ),
        ]


@pytest.mark.asyncio
async def test_rag_search_returns_structured_results() -> None:
    retriever = FakeRetriever()
    tool = RAGSearchTool(retriever)  # type: ignore[arg-type]

    result = await tool.execute(
        {
            "query": "How does enterprise RAG work?",
            "top_k": 2,
        }
    )

    assert retriever.calls == [
        ("How does enterprise RAG work?", 2),
    ]

    assert result["query"] == "How does enterprise RAG work?"
    assert result["retrieved_count"] == 2
    assert result["results"] == [
        {
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "content": "Enterprise RAG retrieves relevant knowledge.",
            "score": 0.95,
            "metadata": {"source": "test"},
        },
        {
            "chunk_id": "chunk-2",
            "document_id": "doc-1",
            "content": "Retrieved context is supplied to the agent.",
            "score": 0.85,
            "metadata": {"source": "test"},
        },
    ]


@pytest.mark.asyncio
async def test_rag_search_defaults_top_k_to_five() -> None:
    retriever = FakeRetriever()
    tool = RAGSearchTool(retriever)  # type: ignore[arg-type]

    await tool.execute({"query": "RAG"})

    assert retriever.calls == [("RAG", 5)]


@pytest.mark.asyncio
async def test_rag_search_rejects_empty_query() -> None:
    retriever = FakeRetriever()
    tool = RAGSearchTool(retriever)  # type: ignore[arg-type]

    with pytest.raises(
        ValueError,
        match="query must be a non-empty string",
    ):
        await tool.execute({"query": "   "})


@pytest.mark.asyncio
async def test_rag_search_rejects_invalid_top_k() -> None:
    retriever = FakeRetriever()
    tool = RAGSearchTool(retriever)  # type: ignore[arg-type]

    with pytest.raises(
        ValueError,
        match="top_k must be between 1 and 10",
    ):
        await tool.execute(
            {
                "query": "RAG",
                "top_k": 11,
            }
        )


def test_rag_search_definition() -> None:
    retriever = FakeRetriever()
    tool = RAGSearchTool(retriever)  # type: ignore[arg-type]

    definition = tool.definition

    assert definition.name == "rag.search"
    assert "knowledge base" in definition.description
    assert definition.input_schema["required"] == ["query"]
    assert definition.input_schema["properties"]["top_k"]["default"] == 5
    assert definition.metadata["category"] == "retrieval"
