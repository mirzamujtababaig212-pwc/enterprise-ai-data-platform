from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from rag.models import Document, DocumentChunk, EmbeddedChunk, RetrievalResult


class DocumentLoader(Protocol):
    """
    Loads source content into canonical Documents.
    """

    async def load(self, source: str) -> Sequence[Document]: ...


class Chunker(Protocol):
    """
    Splits documents into retrievable chunks.
    """

    def chunk(self, document: Document) -> Sequence[DocumentChunk]: ...


class EmbeddingService(Protocol):
    """
    Platform abstraction for generating embeddings.

    Implementations may use the Enterprise LLM Gateway,
    a local model, or another approved provider.
    """

    async def embed(self, text: str) -> Sequence[float]: ...


class VectorStore(Protocol):
    """
    Platform abstraction for vector persistence and similarity search.
    """

    async def upsert(
        self,
        chunks: Sequence[EmbeddedChunk],
    ) -> None: ...

    async def search(
        self,
        embedding: Sequence[float],
        top_k: int = 5,
    ) -> Sequence[RetrievalResult]: ...


class Retriever(Protocol):
    """
    Retrieves relevant document chunks for a query.
    """

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> Sequence[RetrievalResult]: ...
