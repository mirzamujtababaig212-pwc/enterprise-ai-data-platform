from __future__ import annotations
from rag.models import Document, DocumentChunk


class RecursiveChunker:
    """
    Deterministic character-based chunker.

    This is intentionally simple for the first RAG vertical slice.
    Production token-aware and structure-aware chunking can be added
    behind the same Chunker contract later.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 100,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")

        if overlap < 0:
            raise ValueError("overlap must not be negative.")

        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size.")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: Document) -> list[DocumentChunk]:
        content = document.content

        if not content:
            return []

        chunks: list[DocumentChunk] = []

        start = 0
        chunk_index = 0

        step = self.chunk_size - self.overlap

        while start < len(content):
            end = min(start + self.chunk_size, len(content))

            chunks.append(
                DocumentChunk(
                    id=f"{document.id}:chunk:{chunk_index}",
                    document_id=document.id,
                    content=content[start:end],
                    metadata=dict(document.metadata),
                    chunk_index=chunk_index,
                )
            )

            chunk_index += 1

            if end >= len(content):
                break

            start += step

        return chunks
