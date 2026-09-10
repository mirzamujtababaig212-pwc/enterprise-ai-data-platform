from unittest.mock import AsyncMock
from dataclasses import FrozenInstanceError

import pytest

from rag.ingestion import RAGIngestionService, RAGIngestionResult
from rag.models import Document, DocumentChunk, EmbeddedChunk


def _embedded_chunk(
    document_id: str,
    chunk_index: int = 0,
) -> EmbeddedChunk:
    chunk = DocumentChunk(
        id=f"{document_id}:chunk:{chunk_index}",
        document_id=document_id,
        content="Enterprise AI knowledge",
        chunk_index=chunk_index,
    )

    return EmbeddedChunk(
        chunk=chunk,
        embedding=(1.0, 2.0),
    )


@pytest.mark.asyncio
async def test_ingestion_service_indexes_documents_and_returns_metrics():
    indexer = AsyncMock()

    first_document = Document(
        id="doc-1",
        content="First document",
    )

    second_document = Document(
        id="doc-2",
        content="Second document",
    )

    indexer.index.side_effect = [
        [
            _embedded_chunk("doc-1", 0),
            _embedded_chunk("doc-1", 1),
        ],
        [
            _embedded_chunk("doc-2", 0),
        ],
    ]

    service = RAGIngestionService(
        indexer=indexer,
    )

    result = await service.ingest(
        [
            first_document,
            second_document,
        ]
    )

    assert result.documents_processed == 2
    assert result.documents_indexed == 2
    assert result.chunks_created == 3

    assert indexer.index.await_count == 2

    indexer.index.assert_any_await(
        first_document,
    )

    indexer.index.assert_any_await(
        second_document,
    )


@pytest.mark.asyncio
async def test_ingestion_service_handles_empty_documents():
    indexer = AsyncMock()

    service = RAGIngestionService(
        indexer=indexer,
    )

    result = await service.ingest([])

    assert result.documents_processed == 0
    assert result.documents_indexed == 0
    assert result.chunks_created == 0

    indexer.index.assert_not_awaited()


@pytest.mark.asyncio
async def test_ingestion_service_propagates_indexing_failure():
    indexer = AsyncMock()

    document = Document(
        id="failed-document",
        content="Knowledge",
    )

    indexer.index.side_effect = RuntimeError(
        "embedding failed",
    )

    service = RAGIngestionService(
        indexer=indexer,
    )

    with pytest.raises(
        RuntimeError,
        match="embedding failed",
    ):
        await service.ingest(
            [document],
        )


def test_rag_ingestion_result_is_immutable():
    result = RAGIngestionResult(
        documents_processed=1,
        documents_indexed=1,
        chunks_created=2,
    )

    with pytest.raises(FrozenInstanceError):
        result.documents_processed = 2
