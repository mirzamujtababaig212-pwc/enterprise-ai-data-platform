"""Tests for candidate collections."""

from ai_platform.llm_gateway.routing.candidate_set import CandidateSet
from ai_platform.llm_gateway.routing.candidates import RoutingCandidate


def test_candidate_set_preserves_candidates() -> None:
    candidates = CandidateSet(
        [
            RoutingCandidate("openai", "gpt-4o"),
            RoutingCandidate("gemini", "gemini-2.5-pro"),
        ]
    )

    assert len(candidates) == 2
    assert candidates.providers() == ["openai", "gemini"]


def test_candidate_set_sorts_by_priority() -> None:
    candidates = CandidateSet(
        [
            RoutingCandidate(
                "openai",
                "gpt-4o",
                priority=1,
            ),
            RoutingCandidate(
                "azure_openai",
                "gpt-4o",
                priority=10,
            ),
        ]
    )

    result = candidates.sorted_by_priority()

    assert result[0].provider == "azure_openai"
    assert result[1].provider == "openai"


def test_candidate_set_models_are_unique() -> None:
    candidates = CandidateSet(
        [
            RoutingCandidate("openai", "gpt-4o"),
            RoutingCandidate("azure_openai", "gpt-4o"),
            RoutingCandidate("gemini", "gemini-2.5-pro"),
        ]
    )

    assert candidates.models() == [
        "gpt-4o",
        "gemini-2.5-pro",
    ]
