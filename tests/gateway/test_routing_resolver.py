"""Tests for the routing resolver."""

import pytest

from ai_platform.llm_gateway.registry.model_registry import (
    ModelRegistry,
)
from ai_platform.llm_gateway.routing.candidate_set import (
    CandidateSet,
)
from ai_platform.llm_gateway.routing.candidates import (
    RoutingCandidate,
)
from ai_platform.llm_gateway.routing.resolver import (
    ResolvedRoute,
    RoutingResolver,
)


class FakeModelRegistry:
    """Minimal model registry used to test routing behavior."""

    def is_logical_model(
        self,
        model: str,
    ) -> bool:
        return False

    def get_logical_model_routes(
        self,
        model: str,
        capability: str,
    ) -> list:
        return []

    def get_model_capabilities(
        self,
        provider: str,
        model: str,
    ):
        return None

    def get_providers_for_model(
        self,
        capability: str,
        model: str,
    ) -> list[str]:
        if capability == "chat" and model == "gpt-4o":
            return [
                "openai",
                "azure_openai",
            ]

        return []


class FakeProviderResolver:
    """Minimal provider resolver used by resolver tests."""

    def resolve_many(
        self,
        provider_names: list[str],
    ) -> list[object]:
        return [f"provider:{name}" for name in provider_names]


class FakeBalancer:
    """Deterministic fake balancer."""

    def __init__(self) -> None:
        self.calls: list[list[RoutingCandidate]] = []

    def select(
        self,
        candidates: list[RoutingCandidate],
    ) -> RoutingCandidate:
        self.calls.append(candidates)
        return candidates[1]


class FakeProvider:
    """Fake OpenAI-style provider."""

    def supported_chat_models(self) -> list[str]:
        return [
            "gpt-4.1",
            "gpt-4o",
        ]

    def supported_embedding_models(self) -> list[str]:
        return [
            "text-embedding-3-small",
        ]

    async def stream(self, request):
        yield "chunk"


class GeminiProvider:
    """Fake Gemini-style provider."""

    def supported_chat_models(self) -> list[str]:
        return [
            "gemini-2.5-pro",
        ]

    def supported_embedding_models(self) -> list[str]:
        return []


@pytest.fixture
def registry() -> ModelRegistry:
    """Create an isolated model registry."""

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
    registry: ModelRegistry,
) -> RoutingResolver:
    """Create a resolver using the isolated registry."""

    return RoutingResolver(
        model_registry=registry,
    )


def test_resolves_chat_provider(
    resolver: RoutingResolver,
) -> None:
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
    resolver: RoutingResolver,
) -> None:
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
    resolver: RoutingResolver,
) -> None:
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
    resolver: RoutingResolver,
) -> None:
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
    resolver: RoutingResolver,
) -> None:
    names = resolver.resolve_names(
        capability="chat",
        model="gpt-4.1",
    )

    assert names == [
        "openai",
    ]


def test_unknown_model_returns_no_providers(
    resolver: RoutingResolver,
) -> None:
    providers = resolver.resolve(
        capability="chat",
        model="does-not-exist",
    )

    assert providers == []


def test_resolver_builds_candidates_from_registry() -> None:
    resolver = RoutingResolver(
        model_registry=FakeModelRegistry(),
        provider_resolver=FakeProviderResolver(),
        load_balancer=FakeBalancer(),
    )

    candidates = resolver.resolve_candidates(
        capability="chat",
        model="gpt-4o",
    )

    assert isinstance(
        candidates,
        CandidateSet,
    )

    assert [candidate.provider for candidate in candidates] == [
        "openai",
        "azure_openai",
    ]


def test_resolver_selects_balanced_provider() -> None:
    balancer = FakeBalancer()

    resolver = RoutingResolver(
        model_registry=FakeModelRegistry(),
        provider_resolver=FakeProviderResolver(),
        load_balancer=balancer,
    )

    providers = resolver.resolve(
        capability="chat",
        model="gpt-4o",
    )

    assert providers == [
        "provider:azure_openai",
        "provider:openai",
    ]

    assert len(balancer.calls) == 1


def test_explicit_provider_restricts_candidates() -> None:
    resolver = RoutingResolver(
        model_registry=FakeModelRegistry(),
        provider_resolver=FakeProviderResolver(),
        load_balancer=FakeBalancer(),
    )

    candidates = resolver.resolve_candidates(
        capability="chat",
        model="gpt-4o",
        requested_provider="azure_openai",
    )

    assert [candidate.provider for candidate in candidates] == [
        "azure_openai",
    ]


def test_unknown_model_returns_empty_candidates() -> None:
    resolver = RoutingResolver(
        model_registry=FakeModelRegistry(),
        provider_resolver=FakeProviderResolver(),
        load_balancer=FakeBalancer(),
    )

    candidates = resolver.resolve_candidates(
        capability="chat",
        model="does-not-exist",
    )

    assert list(candidates) == []


def test_unknown_provider_returns_empty_candidates() -> None:
    resolver = RoutingResolver(
        model_registry=FakeModelRegistry(),
        provider_resolver=FakeProviderResolver(),
        load_balancer=FakeBalancer(),
    )

    candidates = resolver.resolve_candidates(
        capability="chat",
        model="gpt-4o",
        requested_provider="does-not-exist",
    )

    assert list(candidates) == []


def test_resolver_delegates_candidate_resolution_to_policy() -> None:
    class FakePolicy:
        def __init__(self) -> None:
            self.requests = []

        def resolve_candidates(self, request):
            self.requests.append(request)

            return CandidateSet(
                [
                    RoutingCandidate(
                        provider="openai",
                        model="gpt-4o",
                    ),
                    RoutingCandidate(
                        provider="azure_openai",
                        model="gpt-4o",
                    ),
                ]
            )

    policy = FakePolicy()

    resolver = RoutingResolver(
        model_registry=FakeModelRegistry(),
        routing_policy=policy,
        provider_resolver=FakeProviderResolver(),
        load_balancer=FakeBalancer(),
    )

    candidates = resolver.resolve_candidates(
        capability="chat",
        model="gpt-4o",
    )

    assert [candidate.provider for candidate in candidates] == [
        "openai",
        "azure_openai",
    ]

    assert policy.requests == [
        {
            "capability": "chat",
            "model": "gpt-4o",
            "provider": None,
        }
    ]


def test_resolve_routes_preserves_candidate_models() -> None:
    class FakePolicy:
        def resolve_candidates(self, request):
            return CandidateSet(
                [
                    RoutingCandidate(
                        provider="openai",
                        model="gpt-4.1-mini",
                    ),
                    RoutingCandidate(
                        provider="azure_openai",
                        model="azure-openai-chat",
                    ),
                ]
            )

    resolver = RoutingResolver(
        model_registry=FakeModelRegistry(),
        routing_policy=FakePolicy(),
        provider_resolver=FakeProviderResolver(),
        load_balancer=FakeBalancer(),
    )

    routes = resolver.resolve_routes(
        capability="chat",
        model="enterprise-chat",
    )

    assert all(isinstance(route, ResolvedRoute) for route in routes)

    assert [(route.provider, route.model) for route in routes] == [
        ("provider:azure_openai", "azure-openai-chat"),
        ("provider:openai", "gpt-4.1-mini"),
    ]


def test_resolve_routes_attaches_model_capabilities() -> None:
    from ai_platform.llm_gateway.models.capabilities import ModelCapabilities

    class ProviderWithCapabilities(FakeProvider):
        def model_capabilities(self):
            return {
                "gpt-4.1": ModelCapabilities(
                    context_window_tokens=128000,
                    max_output_tokens=16384,
                    supports_tools=True,
                    supports_vision=True,
                )
            }

    registry = ModelRegistry()

    registry.register_provider(
        "openai",
        ProviderWithCapabilities(),
    )

    resolver = RoutingResolver(
        model_registry=registry,
    )

    routes = resolver.resolve_routes(
        capability="chat",
        model="gpt-4.1",
        requested_provider="openai",
    )

    assert len(routes) == 1
    assert routes[0].model == "gpt-4.1"
    assert routes[0].capabilities is not None
    assert routes[0].capabilities.context_window_tokens == 128000
    assert routes[0].capabilities.max_output_tokens == 16384
    assert routes[0].capabilities.supports_tools is True
    assert routes[0].capabilities.supports_vision is True
