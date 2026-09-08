from typing import Any

import pytest

from ai_platform.llm_gateway.routing.registry_policy import (
    RegistryRoutingPolicy,
)
from ai_platform.llm_gateway.routing.resolver import RoutingResolver
from ai_platform.llm_gateway.routing.router import Router


class FakeGeminiProvider:
    """Fake Gemini provider for deterministic streaming integration tests."""

    name = "gemini"

    def __init__(self) -> None:
        self.stream_calls: list[dict[str, Any]] = []

    async def stream(
        self,
        request: dict[str, Any],
    ):
        self.stream_calls.append(request)

        yield "RAG retrieves relevant context."
        yield " The LLM uses that context to answer the question."


class FakeModelRegistry:
    """Minimal capability-aware registry for Gemini streaming."""

    def get_providers_for_model(
        self,
        capability: str,
        model: str,
    ) -> list[str]:
        if capability == "stream" and model == "gemini-chat":
            return ["gemini"]

        return []


class FakeProviderResolver:
    """Resolve the Gemini test provider."""

    def __init__(
        self,
        provider: FakeGeminiProvider,
    ) -> None:
        self.provider = provider

    def resolve_many(
        self,
        provider_names: list[str],
    ) -> list[FakeGeminiProvider]:
        return [self.provider for provider_name in provider_names if provider_name == "gemini"]


class FakeLoadBalancer:
    """Deterministically select the first candidate."""

    def select(
        self,
        candidates,
    ):
        if not candidates:
            return None

        return candidates[0]


@pytest.mark.asyncio
async def test_stream():
    fake_provider = FakeGeminiProvider()
    registry = FakeModelRegistry()

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

    router = Router(
        routing_resolver=routing_resolver,
    )

    request = {
        "prompt": "Explain RAG.",
        "provider": "gemini",
        "model": "gemini-chat",
        "temperature": 0.7,
        "max_tokens": 100,
        "stream": True,
    }

    chunks = []

    async for chunk in router.route_stream(request):
        chunks.append(chunk)

    assert chunks == [
        "RAG retrieves relevant context.",
        " The LLM uses that context to answer the question.",
    ]

    assert fake_provider.stream_calls == [
        request,
    ]
