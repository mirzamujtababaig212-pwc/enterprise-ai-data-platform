"""Tests for routing load balancing."""

from ai_platform.llm_gateway.routing.balancer import (
    RoundRobinLoadBalancer,
)
from ai_platform.llm_gateway.routing.candidates import RoutingCandidate


def test_round_robin_cycles_candidates() -> None:
    candidates = [
        RoutingCandidate("openai", "gpt-4o"),
        RoutingCandidate("gemini", "gemini-2.5-pro"),
    ]

    balancer = RoundRobinLoadBalancer()

    selections = [balancer.select(candidates).provider for _ in range(4)]

    assert selections == [
        "openai",
        "gemini",
        "openai",
        "gemini",
    ]


def test_weighted_round_robin() -> None:
    candidates = [
        RoutingCandidate(
            "openai",
            "gpt-4o",
            weight=1,
        ),
        RoutingCandidate(
            "azure_openai",
            "gpt-4o",
            weight=2,
        ),
    ]

    balancer = RoundRobinLoadBalancer()

    selections = [balancer.select(candidates).provider for _ in range(6)]

    assert selections == [
        "openai",
        "azure_openai",
        "azure_openai",
        "openai",
        "azure_openai",
        "azure_openai",
    ]


def test_empty_candidates_fail() -> None:
    balancer = RoundRobinLoadBalancer()

    try:
        balancer.select([])
    except ValueError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
