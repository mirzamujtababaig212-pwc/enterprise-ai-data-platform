"""Integration tests for Router and the real routing stack."""

from typing import Any

import pytest

from ai_platform.llm_gateway.routing.registry_policy import (
    RegistryRoutingPolicy,
)
from ai_platform.llm_gateway.routing.resolver import RoutingResolver
from ai_platform.llm_gateway.routing.router import Router


class FakeProvider:
    """Fake provider used to verify real routing-stack execution."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.chat_calls: list[dict[str, Any]] = []
        self.embedding_calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []

    async def chat(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        self.chat_calls.append(request)

        return {
            "provider": self.name,
            "response": "fake-chat-response",
        }

    async def embeddings(
        self,
        request: dict[str, Any],
    ) -> list[float]:
        self.embedding_calls.append(request)

        return [
            0.1,
            0.2,
            0.3,
        ]

    async def stream(
        self,
        request: dict[str, Any],
    ):
        self.stream_calls.append(request)

        yield "fake-stream"


class FakeModelRegistry:
    """Capability-aware fake model registry."""

    def __init__(self) -> None:
        self.providers = {
            (
                "chat",
                "gpt-4o",
            ): [
                "openai",
            ],
            (
                "embeddings",
                "openai-embedding",
            ): [
                "openai",
            ],
            (
                "stream",
                "gpt-4o",
            ): [
                "openai",
            ],
        }

    def get_providers_for_model(
        self,
        capability: str,
        model: str,
    ) -> list[str]:
        return self.providers.get(
            (
                capability,
                model,
            ),
            [],
        )


class FakeProviderResolver:
    """Fake provider resolver matching the production resolver interface."""

    def __init__(self, provider: FakeProvider) -> None:
        self.provider = provider
        self.calls: list[str] = []
        self.many_calls: list[list[str]] = []

    def resolve(self, provider_name: str) -> FakeProvider:
        """Resolve one provider by name."""

        self.calls.append(provider_name)

        if provider_name != self.provider.name:
            raise ValueError(f"Unsupported fake provider: {provider_name}")

        return self.provider

    def resolve_many(
        self,
        provider_names: list[str],
    ) -> list[FakeProvider]:
        """Resolve multiple providers by name."""

        self.many_calls.append(provider_names)

        return [self.resolve(provider_name) for provider_name in provider_names]


class FakeLoadBalancer:
    """Deterministic load balancer for integration tests."""

    def select(
        self,
        candidates,
    ):
        if not candidates:
            return None

        return candidates[0]


@pytest.fixture
def fake_provider() -> FakeProvider:
    """Create a fake OpenAI provider."""

    return FakeProvider("openai")


@pytest.fixture
def registry() -> FakeModelRegistry:
    """Create a capability-aware fake registry."""

    return FakeModelRegistry()


@pytest.fixture
def router(
    registry: FakeModelRegistry,
    fake_provider: FakeProvider,
) -> Router:
    """Create Router using the real routing stack."""

    routing_policy = RegistryRoutingPolicy(
        model_registry=registry,
    )

    routing_resolver = RoutingResolver(
        model_registry=registry,
        routing_policy=routing_policy,
        provider_resolver=FakeProviderResolver(
            fake_provider,
        ),
        load_balancer=FakeLoadBalancer(),
    )

    return Router(
        routing_resolver=routing_resolver,
    )


@pytest.mark.asyncio
async def test_router_uses_real_routing_stack_for_chat(
    router: Router,
    fake_provider: FakeProvider,
) -> None:
    """Router should execute chat through the real routing stack."""

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

    assert response == {
        "provider": "openai",
        "response": "fake-chat-response",
    }

    assert fake_provider.chat_calls == [
        request,
    ]


@pytest.mark.asyncio
async def test_router_uses_real_routing_stack_for_embeddings(
    router: Router,
    fake_provider: FakeProvider,
) -> None:
    """Router should execute embeddings through the real routing stack."""

    request = {
        "provider": "openai",
        "model": "openai-embedding",
        "input": "hello",
    }

    response = await router.route_embeddings(request)

    assert response == [
        0.1,
        0.2,
        0.3,
    ]

    assert fake_provider.embedding_calls == [
        request,
    ]


@pytest.mark.asyncio
async def test_router_uses_real_routing_stack_for_stream(
    router: Router,
    fake_provider: FakeProvider,
) -> None:
    """Router should execute streaming through the real routing stack."""

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

    chunks = []

    async for chunk in router.route_stream(request):
        chunks.append(chunk)

    assert chunks == [
        "fake-stream",
    ]

    assert fake_provider.stream_calls == [
        request,
    ]
