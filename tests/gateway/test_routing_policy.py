import pytest

from ai_platform.llm_gateway.exceptions.gateway_exceptions import (
    ProviderNotFound,
)
from ai_platform.llm_gateway.registry.model_registry import ModelRegistry
from ai_platform.llm_gateway.routing.policies import RoutingPolicy


class FakeProvider:
    def supported_chat_models(self):
        return ["gpt-4.1", "gpt-4o"]

    def supported_embedding_models(self):
        return ["text-embedding-3-small"]

    async def stream(self, request):
        yield "chunk"


class GeminiProvider:
    def supported_chat_models(self):
        return ["gemini-2.5-pro"]

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
def policy(registry):
    return RoutingPolicy(
        model_registry=registry,
    )


def test_explicit_provider_is_primary_candidate(policy):
    candidates = policy.resolve(
        capability="chat",
        model="gpt-4.1",
        requested_provider="openai",
    )

    assert candidates == ["openai"]


def test_explicit_provider_must_support_requested_model(policy):
    with pytest.raises(ValueError, match="Unsupported"):
        policy.resolve(
            capability="chat",
            model="gemini-2.5-pro",
            requested_provider="openai",
        )


def test_explicit_unknown_provider_is_rejected(policy):
    with pytest.raises(ProviderNotFound):
        policy.resolve(
            capability="chat",
            model="gpt-4.1",
            requested_provider="does-not-exist",
        )


def test_discovers_all_matching_providers(policy):
    candidates = policy.resolve(
        capability="chat",
        model="gpt-4.1",
    )

    assert candidates == ["openai"]


def test_discovers_providers_for_same_model(policy, registry):
    registry.register_provider(
        "another-openai-compatible-provider",
        FakeProvider(),
    )

    candidates = policy.resolve(
        capability="chat",
        model="gpt-4.1",
    )

    assert candidates == [
        "openai",
        "another-openai-compatible-provider",
    ]


def test_discovers_embedding_provider(policy):
    candidates = policy.resolve(
        capability="embeddings",
        model="text-embedding-3-small",
    )

    assert candidates == ["openai"]


def test_discovers_stream_using_registry(policy):
    candidates = policy.resolve(
        capability="stream",
        model="gpt-4.1",
    )

    assert candidates == ["openai"]


def test_unknown_model_returns_no_candidates(policy):
    candidates = policy.resolve(
        capability="chat",
        model="does-not-exist",
    )

    assert candidates == []


def test_invalid_capability_returns_no_candidates(policy):
    candidates = policy.resolve(
        capability="unknown",
        model="gpt-4.1",
    )

    assert candidates == []
