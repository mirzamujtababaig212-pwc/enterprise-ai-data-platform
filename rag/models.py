from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    """
    Canonical source document entering the RAG pipeline.
    """

    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentChunk:
    """
    A retrievable segment of a source document.
    """

    id: str
    document_id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_index: int = 0


@dataclass(frozen=True)
class EmbeddedChunk:
    """
    A document chunk together with its embedding vector.
    """

    chunk: DocumentChunk
    embedding: tuple[float, ...]


@dataclass(frozen=True)
class RetrievalResult:
    """
    A retrieved chunk and its similarity score.
    """

    chunk: DocumentChunk
    score: float
