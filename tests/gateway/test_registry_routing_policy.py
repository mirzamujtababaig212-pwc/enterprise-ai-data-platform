"""Tests for registry-backed routing policy."""

from ai_platform.llm_gateway.routing.registry_policy import (
    RegistryRoutingPolicy,
)


class FakeRegistry:
    """Fake model registry."""

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


class FakeLogicalRoute:
    """Fake logical model route."""

    def __init__(
        self,
        provider: str,
        model: str,
    ) -> None:
        self.provider = provider
        self.model = model


class FakeLogicalRegistry:
    """Fake registry with logical model support."""

    def is_logical_model(
        self,
        model: str,
    ) -> bool:
        return model == "enterprise-chat"

    def get_logical_model_routes(
        self,
        model: str,
        capability: str,
    ) -> list[FakeLogicalRoute]:
        if model == "enterprise-chat" and capability == "chat":
            return [
                FakeLogicalRoute(
                    provider="openai",
                    model="gpt-4o",
                ),
                FakeLogicalRoute(
                    provider="azure_openai",
                    model="gpt-4o",
                ),
            ]

        return []

    def get_providers_for_model(
        self,
        capability: str,
        model: str,
    ) -> list[str]:
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


def test_registry_policy_resolves_logical_model_routes() -> None:
    policy = RegistryRoutingPolicy(
        model_registry=FakeLogicalRegistry(),
    )

    candidates = policy.resolve_candidates(
        {
            "capability": "chat",
            "model": "enterprise-chat",
        }
    )

    assert [(candidate.provider, candidate.model) for candidate in candidates] == [
        ("openai", "gpt-4o"),
        ("azure_openai", "gpt-4o"),
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
