from unittest.mock import Mock

from ai_platform.llm_gateway.routing.candidate_set import CandidateSet
from ai_platform.llm_gateway.routing.candidates import RoutingCandidate
from ai_platform.llm_gateway.routing.resolver import RoutingResolver


class FakeRoutingPolicy:
    def __init__(self, candidates):
        self.candidates = candidates

    def resolve_candidates(self, request):
        return CandidateSet(self.candidates)


class FakeProviderResolver:
    def __init__(self):
        self.providers = {
            "provider_a": Mock(name="provider_a"),
            "provider_b": Mock(name="provider_b"),
            "provider_c": Mock(name="provider_c"),
        }

    def resolve_many(self, provider_names):
        return [self.providers[name] for name in provider_names]


class FakeLoadBalancer:
    def __init__(self, selected_provider):
        self.selected_provider = selected_provider

    def select(self, candidates):
        for candidate in candidates:
            if candidate.provider == self.selected_provider:
                return candidate

        raise AssertionError(f"Provider {self.selected_provider} was not found.")


def build_resolver(selected_provider="provider_a"):
    candidates = [
        RoutingCandidate(
            provider="provider_a",
            model="test-model",
            priority=100,
        ),
        RoutingCandidate(
            provider="provider_b",
            model="test-model",
            priority=90,
        ),
        RoutingCandidate(
            provider="provider_c",
            model="test-model",
            priority=80,
        ),
    ]

    provider_resolver = FakeProviderResolver()

    resolver = RoutingResolver(
        routing_policy=FakeRoutingPolicy(candidates),
        provider_resolver=provider_resolver,
        load_balancer=FakeLoadBalancer(selected_provider),
    )

    return resolver, provider_resolver


def test_resolve_returns_full_provider_chain():
    resolver, provider_resolver = build_resolver("provider_a")

    providers = resolver.resolve(
        capability="chat",
        model="test-model",
    )

    assert providers == [
        provider_resolver.providers["provider_a"],
        provider_resolver.providers["provider_b"],
        provider_resolver.providers["provider_c"],
    ]


def test_resolve_rotates_starting_provider():
    resolver, provider_resolver = build_resolver("provider_b")

    providers = resolver.resolve(
        capability="chat",
        model="test-model",
    )

    assert providers == [
        provider_resolver.providers["provider_b"],
        provider_resolver.providers["provider_c"],
        provider_resolver.providers["provider_a"],
    ]


def test_resolve_returns_empty_when_no_candidates():
    resolver = RoutingResolver(
        routing_policy=FakeRoutingPolicy([]),
        provider_resolver=FakeProviderResolver(),
        load_balancer=FakeLoadBalancer("provider_a"),
    )

    providers = resolver.resolve(
        capability="chat",
        model="test-model",
    )

    assert providers == []
