"""Tests for routing candidate models."""

import pytest

from ai_platform.llm_gateway.routing.candidates import RoutingCandidate


def test_routing_candidate_creation() -> None:
    candidate = RoutingCandidate(
        provider="openai",
        model="gpt-4o",
    )

    assert candidate.provider == "openai"
    assert candidate.model == "gpt-4o"
    assert candidate.priority == 0
    assert candidate.weight == 1
    assert candidate.key == "openai:gpt-4o"


def test_routing_candidate_custom_values() -> None:
    candidate = RoutingCandidate(
        provider="azure_openai",
        model="gpt-4o",
        priority=10,
        weight=3,
        metadata={"region": "eastus"},
    )

    assert candidate.priority == 10
    assert candidate.weight == 3
    assert candidate.metadata["region"] == "eastus"


def test_routing_candidate_requires_provider() -> None:
    with pytest.raises(ValueError, match="provider"):
        RoutingCandidate(
            provider="",
            model="gpt-4o",
        )


def test_routing_candidate_requires_model() -> None:
    with pytest.raises(ValueError, match="model"):
        RoutingCandidate(
            provider="openai",
            model="",
        )


def test_routing_candidate_requires_positive_weight() -> None:
    with pytest.raises(ValueError, match="weight"):
        RoutingCandidate(
            provider="openai",
            model="gpt-4o",
            weight=0,
        )
