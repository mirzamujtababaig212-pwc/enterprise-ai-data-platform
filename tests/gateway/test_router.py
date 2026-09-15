from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_platform.llm_gateway.exceptions.gateway_exceptions import (
    ProviderNotFound,
)
from ai_platform.llm_gateway.routing.router import Router


@pytest.fixture
def routing_resolver():
    resolver = MagicMock()
    resolver.is_logical_model.return_value = False
    return resolver


@pytest.fixture
def router(routing_resolver):
    return Router(
        routing_resolver=routing_resolver,
    )


@pytest.fixture
def fake_provider():
    provider = MagicMock()
    provider.name = "openai"

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
async def test_route_chat_uses_physical_model_for_logical_route(
    router,
    routing_resolver,
    fake_provider,
):
    from ai_platform.llm_gateway.routing.resolver import ResolvedRoute

    routing_resolver.is_logical_model.return_value = True
    routing_resolver.resolve_routes.return_value = [
        ResolvedRoute(
            provider=fake_provider,
            model="gpt-4.1-mini",
        )
    ]

    with patch(
        "ai_platform.llm_gateway.routing.router.capability_service.validate_chat",
    ):
        response = await router.route_chat(
            {
                "model": "enterprise-chat",
                "prompt": "Hello",
            }
        )

    assert response["reply"] == "hello"

    routing_resolver.resolve_routes.assert_called_once_with(
        capability="chat",
        model="enterprise-chat",
        requested_provider=None,
    )

    fake_provider.chat.assert_awaited_once_with(
        {
            "model": "gpt-4.1-mini",
            "prompt": "Hello",
        }
    )


@pytest.mark.asyncio
async def test_route_chat_uses_physical_model_for_logical_route_with_provider(
    router,
    routing_resolver,
    fake_provider,
):
    from ai_platform.llm_gateway.routing.resolver import ResolvedRoute

    routing_resolver.is_logical_model.return_value = True
    routing_resolver.resolve_routes.return_value = [
        ResolvedRoute(
            provider=fake_provider,
            model="ollama-chat",
        )
    ]

    with patch(
        "ai_platform.llm_gateway.routing.router.capability_service.validate_chat",
    ) as validate_chat:
        response = await router.route_chat(
            {
                "provider": "ollama",
                "model": "enterprise-chat",
                "prompt": "Hello",
            }
        )

    assert response["reply"] == "hello"

    validate_chat.assert_not_called()

    routing_resolver.resolve_routes.assert_called_once_with(
        capability="chat",
        model="enterprise-chat",
        requested_provider="ollama",
    )

    fake_provider.chat.assert_awaited_once_with(
        {
            "provider": "ollama",
            "model": "ollama-chat",
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
async def test_route_embeddings_with_metadata_preserves_physical_model(
    router,
    routing_resolver,
    fake_provider,
):
    """Physical embedding routes should expose the requested model as metadata."""

    routing_resolver.resolve.return_value = [
        fake_provider,
    ]

    with patch(
        "ai_platform.llm_gateway.routing.router.capability_service.validate_embeddings",
    ):
        result = await router.route_embeddings_with_metadata(
            {
                "provider": "openai",
                "model": "text-embedding-3-small",
            }
        )

    assert result.response == [
        0.1,
        0.2,
        0.3,
    ]
    assert result.provider_name == fake_provider.name
    assert result.model_name == "text-embedding-3-small"

    fake_provider.embeddings.assert_awaited_once_with(
        {
            "provider": "openai",
            "model": "text-embedding-3-small",
        }
    )


@pytest.mark.asyncio
async def test_route_embeddings_uses_physical_model_for_logical_route(
    router,
    routing_resolver,
    fake_provider,
):
    """Logical embedding routes should execute using the physical model."""

    from ai_platform.llm_gateway.routing.resolver import ResolvedRoute

    fake_provider.name = "openai"
    routing_resolver.is_logical_model.return_value = True
    routing_resolver.resolve_routes.return_value = [
        ResolvedRoute(
            provider=fake_provider,
            model="text-embedding-3-small",
        )
    ]

    result = await router.route_embeddings_with_metadata(
        {
            "model": "enterprise-embedding",
        }
    )

    assert result.response == [
        0.1,
        0.2,
        0.3,
    ]
    assert result.provider_name == "openai"
    assert result.model_name == "text-embedding-3-small"

    routing_resolver.resolve_routes.assert_called_once_with(
        capability="embeddings",
        model="enterprise-embedding",
        requested_provider=None,
    )

    fake_provider.embeddings.assert_awaited_once_with(
        {
            "model": "text-embedding-3-small",
        }
    )


@pytest.mark.asyncio
async def test_route_embeddings_fallback_preserves_successful_physical_model(
    routing_resolver,
    fake_provider,
):
    """Fallback metadata should identify the physical model that actually succeeded."""

    from ai_platform.llm_gateway.exceptions.provider_exceptions import (
        ProviderConnectionError,
    )
    from ai_platform.llm_gateway.routing.fallback_executor import FallbackExecutor
    from ai_platform.llm_gateway.routing.resolver import ResolvedRoute

    provider_a = MagicMock()
    provider_a.name = "provider-a"
    provider_a.embeddings = AsyncMock(
        side_effect=ProviderConnectionError("provider-a failure"),
    )

    provider_b = MagicMock()
    provider_b.name = "provider-b"
    provider_b.embeddings = AsyncMock(
        return_value=[0.9, 0.8, 0.7],
    )

    routing_resolver.is_logical_model.return_value = True
    routing_resolver.resolve_routes.return_value = [
        ResolvedRoute(
            provider=provider_a,
            model="provider-a-embedding",
        ),
        ResolvedRoute(
            provider=provider_b,
            model="provider-b-embedding",
        ),
    ]

    fallback_router = Router(
        routing_resolver=routing_resolver,
        fallback_executor=FallbackExecutor(max_retries=0),
    )

    result = await fallback_router.route_embeddings_with_metadata(
        {
            "model": "enterprise-embedding",
        }
    )

    assert result.response == [0.9, 0.8, 0.7]
    assert result.provider_name == "provider-b"
    assert result.model_name == "provider-b-embedding"

    provider_a.embeddings.assert_awaited_once_with(
        {
            "model": "provider-a-embedding",
        }
    )
    provider_b.embeddings.assert_awaited_once_with(
        {
            "model": "provider-b-embedding",
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


@pytest.mark.asyncio
async def test_route_chat_with_bedrock_provider():
    from ai_platform.llm_gateway.config.bedrock_settings import BedrockSettings
    from ai_platform.llm_gateway.providers.bedrock_provider import BedrockProvider

    client = MagicMock()
    client.converse.return_value = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {"text": "Hello from mocked Bedrock."},
                ],
            }
        },
        "usage": {
            "inputTokens": 5,
            "outputTokens": 4,
        },
        "stopReason": "end_turn",
    }

    provider = BedrockProvider(
        client=client,
        settings=BedrockSettings(
            region="us-east-1",
            chat_model="amazon.nova-micro-v1:0",
        ),
    )
    provider.name = "bedrock"

    routing_resolver = MagicMock()
    routing_resolver.is_logical_model.return_value = False
    routing_resolver.resolve.return_value = [provider]

    router = Router(
        routing_resolver=routing_resolver,
    )

    with patch(
        "ai_platform.llm_gateway.routing.router.capability_service.validate_chat",
    ) as validate_chat:
        result = await router.route_chat_with_metadata(
            {
                "provider": "bedrock",
                "model": "bedrock-chat",
                "prompt": "Hello",
            }
        )

    validate_chat.assert_called_once_with(
        "bedrock",
        "bedrock-chat",
    )

    routing_resolver.resolve.assert_called_once_with(
        capability="chat",
        model="bedrock-chat",
        requested_provider="bedrock",
    )

    client.converse.assert_called_once()
    kwargs = client.converse.call_args.kwargs

    assert kwargs["modelId"] == "amazon.nova-micro-v1:0"
    assert kwargs["messages"] == [
        {
            "role": "user",
            "content": [{"text": "Hello"}],
        }
    ]
    assert result.provider_name == "bedrock"
    assert result.model_name == "bedrock-chat"
    assert result.response["reply"] == "Hello from mocked Bedrock."
