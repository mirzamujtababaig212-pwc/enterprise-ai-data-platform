from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_platform.llm_gateway.exceptions.gateway_exceptions import (
    ProviderNotFound,
)
from ai_platform.llm_gateway.routing.router import Router


@pytest.fixture
def routing_resolver():
    return MagicMock()


@pytest.fixture
def router(routing_resolver):
    return Router(
        routing_resolver=routing_resolver,
    )


@pytest.fixture
def fake_provider():
    provider = MagicMock()

    provider.chat = AsyncMock(return_value={"reply": "hello"})

    provider.embeddings = AsyncMock(
        return_value=[
            0.1,
            0.2,
            0.3,
        ]
    )

    provider.health_check = AsyncMock(return_value={"status": "healthy"})

    async def fake_stream(request):
        yield "chunk1"
        yield "chunk2"

    provider.stream = fake_stream

    provider.list_models = AsyncMock(
        return_value=[
            "gpt-4o",
            "gpt-4.1",
        ]
    )

    return provider


@pytest.mark.asyncio
async def test_route_chat(router, routing_resolver, fake_provider):

    routing_resolver.resolve.return_value = [
        fake_provider,
    ]

    with patch(
        "ai_platform.llm_gateway.routing.router.capability_service.validate_chat",
    ):
        response = await router.route_chat(
            {
                "provider": "openai",
                "model": "gpt-4o",
                "prompt": "Hello",
            }
        )

    assert response["reply"] == "hello"

    routing_resolver.resolve.assert_called_once_with(
        capability="chat",
        model="gpt-4o",
        requested_provider="openai",
    )

    fake_provider.chat.assert_awaited_once_with(
        {
            "provider": "openai",
            "model": "gpt-4o",
            "prompt": "Hello",
        }
    )


@pytest.mark.asyncio
async def test_route_embeddings(
    router,
    routing_resolver,
    fake_provider,
):

    routing_resolver.resolve.return_value = [
        fake_provider,
    ]

    with patch(
        "ai_platform.llm_gateway.routing.router.capability_service.validate_embeddings",
    ):
        response = await router.route_embeddings(
            {
                "provider": "openai",
                "model": "openai-embedding",
            }
        )

    assert response == [
        0.1,
        0.2,
        0.3,
    ]

    routing_resolver.resolve.assert_called_once_with(
        capability="embeddings",
        model="openai-embedding",
        requested_provider="openai",
    )

    fake_provider.embeddings.assert_awaited_once_with(
        {
            "provider": "openai",
            "model": "openai-embedding",
        }
    )


@pytest.mark.asyncio
async def test_route_stream(
    router,
    routing_resolver,
    fake_provider,
):

    routing_resolver.resolve.return_value = [
        fake_provider,
    ]

    with patch(
        "ai_platform.llm_gateway.routing.router.capability_service.validate_stream",
    ):
        chunks = []

        async for chunk in router.route_stream(
            {
                "provider": "openai",
                "model": "gpt-4o",
            }
        ):
            chunks.append(chunk)

    assert chunks == [
        "chunk1",
        "chunk2",
    ]

    routing_resolver.resolve.assert_called_once_with(
        capability="stream",
        model="gpt-4o",
        requested_provider="openai",
    )


@pytest.mark.asyncio
async def test_route_health(router, fake_provider):

    with (
        patch(
            "ai_platform.llm_gateway.routing.router.ProviderFactory.list_providers",
            return_value=[
                "openai",
            ],
        ),
        patch(
            "ai_platform.llm_gateway.routing.router.ProviderFactory.get_provider",
            return_value=fake_provider,
        ),
    ):
        response = await router.route_health()

        assert response["openai"]["status"] == "healthy"


@pytest.mark.asyncio
async def test_route_models(router, fake_provider):

    with (
        patch(
            "ai_platform.llm_gateway.routing.router.ProviderFactory.list_providers",
            return_value=[
                "openai",
            ],
        ),
        patch(
            "ai_platform.llm_gateway.routing.router.ProviderFactory.get_provider",
            return_value=fake_provider,
        ),
    ):
        response = await router.route_models()

        assert "gpt-4o" in response["openai"]


@pytest.mark.asyncio
async def test_get_provider_success(router, fake_provider):

    with patch(
        "ai_platform.llm_gateway.routing.router.ProviderFactory.get_provider",
        return_value=fake_provider,
    ):
        provider = await router._get_provider("openai")

        assert provider is fake_provider


@pytest.mark.asyncio
async def test_get_provider_not_found(router):

    with patch(
        "ai_platform.llm_gateway.routing.router.ProviderFactory.get_provider",
        return_value=None,
    ):
        with pytest.raises(ProviderNotFound):
            await router._get_provider("bad-provider")


@pytest.mark.asyncio
async def test_route_chat_no_provider_supports_model(
    router,
    routing_resolver,
):

    routing_resolver.resolve.return_value = []

    with (
        patch(
            "ai_platform.llm_gateway.routing.router.capability_service.validate_chat",
        ),
        pytest.raises(
            ProviderNotFound,
            match="No provider supports chat model",
        ),
    ):
        await router.route_chat(
            {
                "model": "does-not-exist",
                "prompt": "Hello",
            }
        )


@pytest.mark.asyncio
async def test_route_chat_discovers_provider_when_not_requested(
    router,
    routing_resolver,
    fake_provider,
):

    routing_resolver.resolve.return_value = [
        fake_provider,
    ]

    with patch(
        "ai_platform.llm_gateway.routing.router.capability_service.validate_chat",
    ):
        response = await router.route_chat(
            {
                "model": "gpt-4o",
                "prompt": "Hello",
            }
        )

    assert response["reply"] == "hello"

    routing_resolver.resolve.assert_called_once_with(
        capability="chat",
        model="gpt-4o",
        requested_provider=None,
    )
