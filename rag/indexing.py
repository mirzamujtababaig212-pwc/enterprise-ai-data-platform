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

        return embedded_chunks
