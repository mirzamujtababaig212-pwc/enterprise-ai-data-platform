import pytest

from ai_platform.llm_gateway.registry.model_registry import (
    ModelRegistry,
)
from ai_platform.llm_gateway.routing.resolver import (
    RoutingResolver,
)


class FakeProvider:
    def supported_chat_models(self):
        return [
            "gpt-4.1",
            "gpt-4o",
        ]

    def supported_embedding_models(self):
        return [
            "text-embedding-3-small",
        ]

    async def stream(self, request):
        yield "chunk"


class GeminiProvider:
    def supported_chat_models(self):
        return [
            "gemini-2.5-pro",
        ]

    def supported_embedding_models(self):
        return []


@pytest.fixture
def registry():
    registry = ModelRegistry()

    registry.register_provider(
        "openai",
        FakeProvider(),
    )

    registry.register_provider(
        "gemini",
        GeminiProvider(),
    )

    return registry


@pytest.fixture
def resolver(
    registry,
):
    return RoutingResolver(
        model_registry=registry,
    )


def test_resolves_chat_provider(
    resolver,
):
    providers = resolver.resolve(
        capability="chat",
        model="gpt-4.1",
    )

    assert len(providers) == 1
    assert isinstance(
        providers[0],
        FakeProvider,
    )


def test_resolves_embedding_provider(
    resolver,
):
    providers = resolver.resolve(
        capability="embeddings",
        model="text-embedding-3-small",
    )

    assert len(providers) == 1
    assert isinstance(
        providers[0],
        FakeProvider,
    )


def test_resolves_stream_provider(
    resolver,
):
    providers = resolver.resolve(
        capability="stream",
        model="gpt-4.1",
    )

    assert len(providers) == 1
    assert isinstance(
        providers[0],
        FakeProvider,
    )


def test_explicit_provider(
    resolver,
):
    providers = resolver.resolve(
        capability="chat",
        model="gpt-4.1",
        requested_provider="openai",
    )

    assert len(providers) == 1
    assert isinstance(
        providers[0],
        FakeProvider,
    )


def test_resolve_names(
    resolver,
):
    names = resolver.resolve_names(
        capability="chat",
        model="gpt-4.1",
    )

    assert names == [
        "openai",
    ]


def test_unknown_model_returns_no_providers(
    resolver,
):
    providers = resolver.resolve(
        capability="chat",
        model="does-not-exist",
    )

    assert providers == []
