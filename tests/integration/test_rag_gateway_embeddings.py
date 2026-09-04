import pytest

from ai_platform.llm_gateway.routing.router import Router
from rag.embeddings import GatewayEmbeddingService


@pytest.mark.asyncio
async def test_rag_embedding_service_uses_mock_gateway_provider():
    service = GatewayEmbeddingService(
        provider="mock",
        model="mock-embedding",
        gateway_router=Router(),
    )

    result = await service.embed("Enterprise AI platform")

    assert isinstance(result, list)
    assert len(result) == 4

    assert all(isinstance(value, float) for value in result)


@pytest.mark.asyncio
async def test_rag_embedding_service_is_deterministic_with_mock_provider():
    service = GatewayEmbeddingService(
        provider="mock",
        model="mock-embedding",
        gateway_router=Router(),
    )

    first = await service.embed("Enterprise AI platform")

    second = await service.embed("Enterprise AI platform")

    assert first == second
