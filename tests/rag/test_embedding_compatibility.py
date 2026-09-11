import pytest

from rag.compatibility import (
    EmbeddingCompatibilityError,
    EmbeddingCompatibilityPolicy,
)
from rag.models import EmbeddingIdentity


def identity(
    *,
    provider: str = "openai",
    model: str = "text-embedding-3-small",
    dimension: int = 1536,
) -> EmbeddingIdentity:
    return EmbeddingIdentity(
        requested_provider=provider,
        requested_model=model,
        resolved_provider=provider,
        resolved_model=model,
        dimension=dimension,
    )


def test_matching_embedding_identities_are_compatible():
    query = identity()
    stored = identity()

    EmbeddingCompatibilityPolicy.validate(
        query=query,
        stored=stored,
    )


def test_different_embedding_dimensions_are_rejected():
    query = identity(dimension=1536)
    stored = identity(dimension=3072)

    with pytest.raises(
        EmbeddingCompatibilityError,
        match="dimensions",
    ):
        EmbeddingCompatibilityPolicy.validate(
            query=query,
            stored=stored,
        )


def test_different_embedding_providers_are_rejected():
    query = identity(provider="openai")
    stored = identity(provider="azure_openai")

    with pytest.raises(
        EmbeddingCompatibilityError,
        match="providers",
    ):
        EmbeddingCompatibilityPolicy.validate(
            query=query,
            stored=stored,
        )


def test_different_embedding_models_are_rejected_even_when_dimensions_match():
    query = identity(
        model="embedding-model-a",
        dimension=1536,
    )

    stored = identity(
        model="embedding-model-b",
        dimension=1536,
    )

    with pytest.raises(
        EmbeddingCompatibilityError,
        match="models",
    ):
        EmbeddingCompatibilityPolicy.validate(
            query=query,
            stored=stored,
        )


def test_same_requested_model_but_different_resolved_models_are_rejected():
    query = EmbeddingIdentity(
        requested_provider="enterprise",
        requested_model="enterprise-embedding",
        resolved_provider="openai",
        resolved_model="text-embedding-3-small",
        dimension=1536,
    )

    stored = EmbeddingIdentity(
        requested_provider="enterprise",
        requested_model="enterprise-embedding",
        resolved_provider="openai",
        resolved_model="text-embedding-3-large",
        dimension=1536,
    )

    with pytest.raises(
        EmbeddingCompatibilityError,
        match="models",
    ):
        EmbeddingCompatibilityPolicy.validate(
            query=query,
            stored=stored,
        )


def test_same_requested_model_but_different_resolved_providers_are_rejected():
    query = EmbeddingIdentity(
        requested_provider="enterprise",
        requested_model="enterprise-embedding",
        resolved_provider="openai",
        resolved_model="text-embedding-3-small",
        dimension=1536,
    )

    stored = EmbeddingIdentity(
        requested_provider="enterprise",
        requested_model="enterprise-embedding",
        resolved_provider="azure_openai",
        resolved_model="text-embedding-3-small",
        dimension=1536,
    )

    with pytest.raises(
        EmbeddingCompatibilityError,
        match="providers",
    ):
        EmbeddingCompatibilityPolicy.validate(
            query=query,
            stored=stored,
        )
