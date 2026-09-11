from __future__ import annotations

from rag.models import EmbeddingIdentity


class EmbeddingCompatibilityError(ValueError):
    """
    Raised when query and stored embeddings belong to incompatible
    embedding spaces.
    """


class EmbeddingCompatibilityPolicy:
    """
    Defines whether two embedding identities are safe to compare.

    Enterprise RAG retrieval must not rely only on vector dimension.
    Different embedding models can produce vectors with the same
    dimension while representing completely different semantic spaces.
    """

    @staticmethod
    def validate(
        query: EmbeddingIdentity,
        stored: EmbeddingIdentity,
    ) -> None:
        if query.dimension != stored.dimension:
            raise EmbeddingCompatibilityError(
                "Embedding dimensions are incompatible: "
                f"query={query.dimension}, "
                f"stored={stored.dimension}."
            )

        if query.resolved_provider != stored.resolved_provider:
            raise EmbeddingCompatibilityError(
                "Embedding providers are incompatible: "
                f"query={query.resolved_provider!r}, "
                f"stored={stored.resolved_provider!r}."
            )

        if query.resolved_model != stored.resolved_model:
            raise EmbeddingCompatibilityError(
                "Embedding models are incompatible: "
                f"query={query.resolved_model!r}, "
                f"stored={stored.resolved_model!r}."
            )
