from unittest.mock import AsyncMock

import pytest

from rag.embeddings import GatewayEmbeddingService


@pytest.mark.asyncio
async def test_gateway_embedding_service_delegates_to_gateway():
    gateway_router = AsyncMock()

    gateway_router.route_embeddings.return_value = [
        0.1,
        0.2,
        0.3,
        0.4,
    ]

    service = GatewayEmbeddingService(
        provider="mock",
        model="mock-embedding",
        gateway_router=gateway_router,
    )

    result = await service.embed("hello world")

    assert result == [
        0.1,
        0.2,
        0.3,
        0.4,
    ]

    gateway_router.route_embeddings.assert_awaited_once_with(
        {
            "provider": "mock",
            "model": "mock-embedding",
            "text": "hello world",
        }
    )


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
