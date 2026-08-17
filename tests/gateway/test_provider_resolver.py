import pytest

from ai_platform.llm_gateway.exceptions.gateway_exceptions import (
    ProviderNotFound,
)
from ai_platform.llm_gateway.providers.resolver import (
    ProviderResolver,
)
from ai_platform.llm_gateway.registry.model_registry import (
    ModelRegistry,
)


class FakeProvider:
    def supported_chat_models(self):
        return [
            "gpt-4.1",
        ]

    def supported_embedding_models(self):
        return []


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
    return ProviderResolver(
        model_registry=registry,
    )


def test_resolves_provider(
    resolver,
):
    provider = resolver.resolve(
        "openai",
    )

    assert isinstance(
        provider,
        FakeProvider,
    )


def test_resolves_gemini_provider(
    resolver,
):
    provider = resolver.resolve(
        "gemini",
    )

    assert isinstance(
        provider,
        GeminiProvider,
    )


def test_unknown_provider_raises_provider_not_found(
    resolver,
):
    with pytest.raises(
        ProviderNotFound,
        match="Unsupported provider",
    ):
        resolver.resolve(
            "does-not-exist",
        )


def test_resolve_many_preserves_order(
    resolver,
):
    providers = resolver.resolve_many(
        [
            "gemini",
            "openai",
        ]
    )

    assert isinstance(
        providers[0],
        GeminiProvider,
    )

    assert isinstance(
        providers[1],
        FakeProvider,
    )


def test_resolve_many_fails_for_unknown_provider(
    resolver,
):
    with pytest.raises(
        ProviderNotFound,
    ):
        resolver.resolve_many(
            [
                "openai",
                "does-not-exist",
            ]
        )
