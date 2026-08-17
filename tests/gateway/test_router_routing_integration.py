"""Tests for Router integration with the routing resolver."""

from typing import Any

import pytest

from ai_platform.llm_gateway.exceptions.gateway_exceptions import ProviderNotFound
from ai_platform.llm_gateway.routing.router import Router


class FakeProvider:
    """Fake provider used to verify router execution."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.chat_calls: list[dict[str, Any]] = []
        self.embedding_calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []

    async def chat(self, request: dict[str, Any]) -> dict[str, Any]:
        """Record and return a fake chat response."""

        self.chat_calls.append(request)

        return {
            "provider": self.name,
            "response": "fake-response",
        }

    async def embeddings(
        self,
        request: dict[str, Any],
    ) -> list[float]:
        """Record and return a fake embedding response."""

        self.embedding_calls.append(request)

        return [0.1, 0.2, 0.3]

    async def stream(
        self,
        request: dict[str, Any],
    ):
        """Record and return a fake streaming response."""

        self.stream_calls.append(request)

        yield "fake-stream"


class FakeRoutingResolver:
    """Fake routing resolver used to verify router delegation."""

    def __init__(self, provider: FakeProvider) -> None:
        self.provider = provider
        self.calls: list[dict[str, Any]] = []

    def resolve(
        self,
        capability: str,
        model: str,
        requested_provider: str | None = None,
    ) -> list[FakeProvider]:
        """Record routing information and return the selected provider."""

        self.calls.append(
            {
                "capability": capability,
                "model": model,
                "requested_provider": requested_provider,
            }
        )

        return [self.provider]


@pytest.fixture
def fake_provider() -> FakeProvider:
    """Create a fake provider."""

    return FakeProvider("openai")


@pytest.fixture
def routing_resolver(
    fake_provider: FakeProvider,
) -> FakeRoutingResolver:
    """Create a fake routing resolver."""

    return FakeRoutingResolver(fake_provider)


@pytest.fixture
def router(
    routing_resolver: FakeRoutingResolver,
) -> Router:
    """Create a router using the fake routing resolver."""

    return Router(
        routing_resolver=routing_resolver,
    )


@pytest.mark.asyncio
async def test_router_delegates_chat_routing(
    router: Router,
    routing_resolver: FakeRoutingResolver,
    fake_provider: FakeProvider,
) -> None:
    """Router should delegate chat provider selection."""

    request = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": "hello",
            }
        ],
    }

    response = await router.route_chat(request)

    assert response == {
        "provider": "openai",
        "response": "fake-response",
    }

    assert routing_resolver.calls == [
        {
            "capability": "chat",
            "model": "gpt-4o",
            "requested_provider": None,
        }
    ]

    assert fake_provider.chat_calls == [request]


@pytest.mark.asyncio
async def test_router_passes_explicit_provider_to_resolver(
    router: Router,
    routing_resolver: FakeRoutingResolver,
    fake_provider: FakeProvider,
) -> None:
    """Router should forward an explicitly requested provider."""

    request = {
        "provider": "openai",
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": "hello",
            }
        ],
    }

    response = await router.route_chat(request)

    assert response["provider"] == "openai"

    assert routing_resolver.calls == [
        {
            "capability": "chat",
            "model": "gpt-4o",
            "requested_provider": "openai",
        }
    ]

    assert fake_provider.chat_calls == [request]


@pytest.mark.asyncio
async def test_router_delegates_embedding_routing(
    router: Router,
    routing_resolver: FakeRoutingResolver,
    fake_provider: FakeProvider,
) -> None:
    """Router should delegate embedding provider selection."""

    request = {
        "model": "text-embedding-3-small",
        "input": "hello",
    }

    response = await router.route_embeddings(request)

    assert response == [
        0.1,
        0.2,
        0.3,
    ]

    assert routing_resolver.calls == [
        {
            "capability": "embeddings",
            "model": "text-embedding-3-small",
            "requested_provider": None,
        }
    ]

    assert fake_provider.embedding_calls == [request]


@pytest.mark.asyncio
async def test_router_delegates_stream_routing(
    router: Router,
    routing_resolver: FakeRoutingResolver,
    fake_provider: FakeProvider,
) -> None:
    """Router should delegate streaming provider selection."""

    request = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": "hello",
            }
        ],
    }

    chunks = []

    async for chunk in router.route_stream(request):
        chunks.append(chunk)

    assert chunks == [
        "fake-stream",
    ]

    assert routing_resolver.calls == [
        {
            "capability": "stream",
            "model": "gpt-4o",
            "requested_provider": None,
        }
    ]

    assert fake_provider.stream_calls == [request]


@pytest.mark.asyncio
async def test_router_raises_when_resolver_returns_no_provider() -> None:
    """Router should reject requests with no routable provider."""

    class EmptyRoutingResolver:
        def resolve(
            self,
            capability: str,
            model: str,
            requested_provider: str | None = None,
        ) -> list[Any]:
            return []

    router = Router(
        routing_resolver=EmptyRoutingResolver(),
    )

    request = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": "hello",
            }
        ],
    }

    with pytest.raises(
        ProviderNotFound,
        match="No provider supports chat model",
    ):
        await router.route_chat(request)
