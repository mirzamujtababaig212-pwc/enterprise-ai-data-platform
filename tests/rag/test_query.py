from unittest.mock import AsyncMock, MagicMock

import pytest

from rag.models import DocumentChunk, RetrievalResult
from rag.query import RAGQueryService


@pytest.mark.asyncio
async def test_rag_query_returns_answer_and_sources():

    retrieved = [
        RetrievalResult(
            chunk=DocumentChunk(
                id="doc-1:chunk:0",
                document_id="doc-1",
                content="The enterprise platform supports RAG.",
                metadata={
                    "source": "architecture.md",
                },
                chunk_index=0,
            ),
            score=0.95,
        ),
        RetrievalResult(
            chunk=DocumentChunk(
                id="doc-2:chunk:0",
                document_id="doc-2",
                content="The platform provides model routing.",
                metadata={
                    "source": "gateway.md",
                },
                chunk_index=0,
            ),
            score=0.81,
        ),
    ]

    retriever = MagicMock()
    retriever.retrieve = AsyncMock(
        return_value=retrieved,
    )

    chat_service = MagicMock()
    chat_service.generate = AsyncMock(
        return_value={
            "reply": "The platform supports RAG and model routing.",
        }
    )

    service = RAGQueryService(
        retriever=retriever,
        chat_service=chat_service,
    )

    result = await service.query(
        "What does the platform support?",
        top_k=2,
    )

    assert result.answer == ("The platform supports RAG and model routing.")

    assert result.retrieved_count == 2

    assert len(result.sources) == 2

    assert result.sources[0].document_id == "doc-1"
    assert result.sources[0].chunk_id == "doc-1:chunk:0"
    assert result.sources[0].score == 0.95

    retriever.retrieve.assert_awaited_once_with(
        "What does the platform support?",
        top_k=2,
    )

    chat_service.generate.assert_awaited_once()

    generated_prompt = chat_service.generate.call_args.args[0]

    assert "What does the platform support?" in generated_prompt
    assert "The enterprise platform supports RAG." in generated_prompt
    assert "The platform provides model routing." in generated_prompt
