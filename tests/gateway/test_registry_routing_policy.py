"""Tests for registry-backed routing policy."""

from ai_platform.llm_gateway.routing.registry_policy import (
    RegistryRoutingPolicy,
)


class FakeRegistry:
    """Fake model registry."""

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


def test_registry_policy_resolves_candidates() -> None:
    policy = RegistryRoutingPolicy(
        model_registry=FakeRegistry(),
    )

    candidates = policy.resolve_candidates(
        {
            "capability": "chat",
            "model": "gpt-4o",
        }
    )

    assert [candidate.provider for candidate in candidates] == [
        "openai",
        "azure_openai",
    ]


def test_registry_policy_filters_explicit_provider() -> None:
    policy = RegistryRoutingPolicy(
        model_registry=FakeRegistry(),
    )

    candidates = policy.resolve_candidates(
        {
            "capability": "chat",
            "model": "gpt-4o",
            "provider": "azure_openai",
        }
    )

    assert [candidate.provider for candidate in candidates] == [
        "azure_openai",
    ]


def test_registry_policy_returns_empty_for_unknown_model() -> None:
    policy = RegistryRoutingPolicy(
        model_registry=FakeRegistry(),
    )

    candidates = policy.resolve_candidates(
        {
            "capability": "chat",
            "model": "does-not-exist",
        }
    )

    assert list(candidates) == []


def test_registry_policy_returns_empty_for_missing_model() -> None:
    policy = RegistryRoutingPolicy(
        model_registry=FakeRegistry(),
    )

    candidates = policy.resolve_candidates(
        {
            "capability": "chat",
        }
    )

    assert list(candidates) == []


def test_registry_policy_returns_empty_for_missing_capability() -> None:
    policy = RegistryRoutingPolicy(
        model_registry=FakeRegistry(),
    )

    candidates = policy.resolve_candidates(
        {
            "model": "gpt-4o",
        }
    )

    assert list(candidates) == []
