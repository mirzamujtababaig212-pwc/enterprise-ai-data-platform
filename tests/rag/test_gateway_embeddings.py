from unittest.mock import AsyncMock

import pytest

from rag.embeddings import GatewayEmbeddingService


@pytest.mark.asyncio
async def test_gateway_embedding_service_delegates_to_gateway():
    gateway_router = AsyncMock()

    gateway_router.route_embeddings_with_metadata.return_value = type(
        "EmbeddingResult",
        (),
        {
            "response": [0.1, 0.2, 0.3, 0.4],
            "provider_name": "mock",
            "model_name": "mock-embedding",
        },
    )()

    service = GatewayEmbeddingService(
        provider="mock",
        model="mock-embedding",
        gateway_router=gateway_router,
    )

    result = await service.embed("hello world")

    assert isinstance(result, list)
    assert result == [0.1, 0.2, 0.3, 0.4]

    gateway_router.route_embeddings_with_metadata.assert_awaited_once_with(
        {
            "provider": "mock",
            "model": "mock-embedding",
            "text": "hello world",
        }
    )


@pytest.mark.asyncio
async def test_gateway_embedding_service_returns_embedding_metadata():
    gateway_router = AsyncMock()

    gateway_router.route_embeddings_with_metadata.return_value = type(
        "EmbeddingResult",
        (),
        {
            "response": [0.1, 0.2, 0.3, 0.4],
            "provider_name": "openai",
            "model_name": "text-embedding-3-small",
        },
    )()

    service = GatewayEmbeddingService(
        provider="openai",
        model="openai-embedding",
        gateway_router=gateway_router,
    )

    result = await service.embed_with_metadata("hello world")

    assert result.vector == (0.1, 0.2, 0.3, 0.4)

    assert result.identity.requested_provider == "openai"
    assert result.identity.requested_model == "openai-embedding"
    assert result.identity.resolved_provider == "openai"
    assert result.identity.resolved_model == "text-embedding-3-small"
    assert result.identity.dimension == 4


@pytest.mark.asyncio
async def test_gateway_embedding_service_preserves_fallback_provenance():
    gateway_router = AsyncMock()

    gateway_router.route_embeddings_with_metadata.return_value = type(
        "EmbeddingResult",
        (),
        {
            "response": [0.9, 0.8, 0.7],
            "provider_name": "provider-b",
            "model_name": "provider-b-embedding",
        },
    )()

    service = GatewayEmbeddingService(
        provider="enterprise",
        model="enterprise-embedding",
        gateway_router=gateway_router,
    )

    result = await service.embed_with_metadata("hello world")

    assert result.identity.requested_provider == "enterprise"
    assert result.identity.requested_model == "enterprise-embedding"
    assert result.identity.resolved_provider == "provider-b"
    assert result.identity.resolved_model == "provider-b-embedding"
    assert result.identity.dimension == 3


@pytest.mark.asyncio
async def test_gateway_embedding_service_rejects_empty_text():
    gateway_router = AsyncMock()

    service = GatewayEmbeddingService(
        provider="mock",
        model="mock-embedding",
        gateway_router=gateway_router,
    )

    with pytest.raises(ValueError, match="empty"):
        await service.embed("")


@pytest.mark.asyncio
async def test_gateway_embedding_service_rejects_whitespace_text():
    gateway_router = AsyncMock()

    service = GatewayEmbeddingService(
        provider="mock",
        model="mock-embedding",
        gateway_router=gateway_router,
    )

    with pytest.raises(ValueError, match="empty"):
        await service.embed("   ")


def test_gateway_embedding_service_rejects_empty_provider():
    with pytest.raises(ValueError, match="provider"):
        GatewayEmbeddingService(
            provider="",
            model="mock-embedding",
        )


def test_gateway_embedding_service_rejects_empty_model():
    with pytest.raises(ValueError, match="model"):
        GatewayEmbeddingService(
            provider="mock",
            model="",
        )
