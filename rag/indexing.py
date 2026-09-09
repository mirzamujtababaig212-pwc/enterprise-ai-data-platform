from __future__ import annotations

from collections.abc import Sequence

from rag.contracts import Chunker, EmbeddingService, VectorStore
from rag.models import Document, EmbeddedChunk


class RAGIndexer:
    """
    Indexes source documents into a vector store.

    Pipeline:

        Document
            ↓
        Chunker
            ↓
        EmbeddingService
            ↓
        VectorStore
    """

    def __init__(
        self,
        chunker: Chunker,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
    ) -> None:
        self.chunker = chunker
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    async def index(
        self,
        document: Document,
        *,
        previous_chunk_ids: Sequence[str] = (),
    ) -> Sequence[EmbeddedChunk]:
        chunks = self.chunker.chunk(document)

        embedded_chunks: list[EmbeddedChunk] = []

        for chunk in chunks:
            embedding = await self.embedding_service.embed(chunk.content)

            embedded_chunks.append(
                EmbeddedChunk(
                    chunk=chunk,
                    embedding=tuple(float(value) for value in embedding),
                )
            )

        if embedded_chunks:
            await self.vector_store.upsert(embedded_chunks)

        new_chunk_ids = {embedded_chunk.chunk.id for embedded_chunk in embedded_chunks}

        stale_chunk_ids = [
            chunk_id for chunk_id in previous_chunk_ids if chunk_id not in new_chunk_ids
        ]

        if stale_chunk_ids:
            await self.vector_store.delete_chunks(stale_chunk_ids)

        return embedded_chunks
