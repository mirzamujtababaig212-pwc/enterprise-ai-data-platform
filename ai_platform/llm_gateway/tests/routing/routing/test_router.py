from unittest.mock import AsyncMock, Mock

import pytest

from ai_platform.llm_gateway.routing.fallback_executor import FallbackResult
from ai_platform.llm_gateway.routing.resolver import ResolvedRoute, RoutingResolver
from ai_platform.llm_gateway.routing.router import Router


@pytest.mark.asyncio
async def test_logical_model_without_provider_allows_automatic_routing():
    openai_provider = Mock()
    openai_provider.name = "openai"

    ollama_provider = Mock()
    ollama_provider.name = "ollama"

    resolver = Mock(spec=RoutingResolver)
    resolver.is_logical_model.return_value = True
    resolver.resolve_routes.return_value = [
        ResolvedRoute(
            provider=openai_provider,
            model="gpt-4.1-mini",
        ),
        ResolvedRoute(
            provider=ollama_provider,
            model="ollama-chat",
        ),
    ]

    fallback_executor = Mock()
    fallback_executor.execute = AsyncMock(
        return_value=FallbackResult(
            response={"reply": "automatic routing works"},
            provider_name="ollama",
            attempts=(),
        )
    )

    router = Router(
        routing_resolver=resolver,
        fallback_executor=fallback_executor,
    )

    result = await router.route_chat_with_metadata(
        {
            "model": "enterprise-chat",
            "prompt": "test",
            "provider": None,
        }
    )

    resolver.resolve_routes.assert_called_once_with(
        capability="chat",
        model="enterprise-chat",
        requested_provider=None,
    )

    providers = fallback_executor.execute.await_args.args[0]

    assert providers == [
        openai_provider,
        ollama_provider,
    ]
    assert result.provider_name == "ollama"
    assert result.response == {"reply": "automatic routing works"}
    assert result.model_name == "ollama-chat"


@pytest.mark.asyncio
async def test_logical_model_with_explicit_provider_remains_constrained():
    ollama_provider = Mock()
    ollama_provider.name = "ollama"

    resolver = Mock(spec=RoutingResolver)
    resolver.is_logical_model.return_value = True
    resolver.resolve_routes.return_value = [
        ResolvedRoute(
            provider=ollama_provider,
            model="ollama-chat",
        ),
    ]

    fallback_executor = Mock()
    fallback_executor.execute = AsyncMock(
        return_value=FallbackResult(
            response={"reply": "ollama routing works"},
            provider_name="ollama",
            attempts=(),
        )
    )

    router = Router(
        routing_resolver=resolver,
        fallback_executor=fallback_executor,
    )

    result = await router.route_chat_with_metadata(
        {
            "model": "enterprise-chat",
            "prompt": "test",
            "provider": "ollama",
        }
    )

    resolver.resolve_routes.assert_called_once_with(
        capability="chat",
        model="enterprise-chat",
        requested_provider="ollama",
    )

    providers = fallback_executor.execute.await_args.args[0]

    assert providers == [ollama_provider]
    assert result.provider_name == "ollama"
    assert result.response == {"reply": "ollama routing works"}
    assert result.model_name == "ollama-chat"


@pytest.mark.asyncio
async def test_physical_model_preserves_requested_model_name():
    openai_provider = Mock()
    openai_provider.name = "openai"

    resolver = Mock(spec=RoutingResolver)
    resolver.is_logical_model.return_value = False
    resolver.resolve.return_value = [openai_provider]

    fallback_executor = Mock()
    fallback_executor.execute = AsyncMock(
        return_value=FallbackResult(
            response={"reply": "physical routing works"},
            provider_name="openai",
            attempts=(),
        )
    )

    router = Router(
        routing_resolver=resolver,
        fallback_executor=fallback_executor,
    )

    result = await router.route_chat_with_metadata(
        {
            "model": "gpt-4.1-mini",
            "prompt": "test",
            "provider": "openai",
        }
    )

    assert result.provider_name == "openai"
    assert result.model_name == "gpt-4.1-mini"
