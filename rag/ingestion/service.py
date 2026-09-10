from __future__ import annotations

from collections.abc import Sequence

from rag.indexing import RAGIndexer
from rag.ingestion.models import RAGIngestionResult
from rag.models import Document


class RAGIngestionService:
    """
    Orchestrates ingestion of canonical Documents into the RAG index.

    Responsibilities:
    - process canonical Documents
    - delegate chunking, embedding, and persistence to RAGIndexer
    - return ingestion metrics

    The service deliberately does not own source loading, chunking,
    embedding, or vector-store implementation details.
    """

    def __init__(
        self,
        indexer: RAGIndexer,
    ) -> None:
        self._indexer = indexer

    async def ingest(
        self,
        documents: Sequence[Document],
    ) -> RAGIngestionResult:
        documents_processed = 0
        documents_indexed = 0
        chunks_created = 0

        for document in documents:
            documents_processed += 1

            embedded_chunks = await self._indexer.index(
                document,
            )

            documents_indexed += 1
            chunks_created += len(embedded_chunks)

        return RAGIngestionResult(
            documents_processed=documents_processed,
            documents_indexed=documents_indexed,
            chunks_created=chunks_created,
        )
