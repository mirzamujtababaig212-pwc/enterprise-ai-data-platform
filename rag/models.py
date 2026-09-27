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
class EmbeddingIdentity:
    """
    Identity of the embedding model that actually generated a vector.
    """

    requested_provider: str | None
    requested_model: str
    resolved_provider: str
    resolved_model: str
    dimension: int


@dataclass(frozen=True)
class EmbeddingResult:
    """
    An embedding vector together with its resolved model identity.
    """

    vector: tuple[float, ...]
    identity: EmbeddingIdentity


@dataclass(frozen=True)
class EmbeddedChunk:
    """
    A document chunk together with its embedding vector.
    """

    chunk: DocumentChunk
    embedding: tuple[float, ...]
    embedding_identity: EmbeddingIdentity | None = None


@dataclass(frozen=True)
class RetrievalResult:
    """
    A retrieved chunk and its ranking scores.

    ``score`` remains the final ranking score for backward compatibility.
    ``retrieval_score`` preserves the original retriever score when a
    reranker is involved, while ``reranker_score`` records the reranker's
    score independently.
    """

    chunk: DocumentChunk
    score: float
    embedding_identity: EmbeddingIdentity | None = None
    retrieval_score: float | None = None
    reranker_score: float | None = None
