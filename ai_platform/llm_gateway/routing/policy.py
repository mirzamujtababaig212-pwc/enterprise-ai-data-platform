"""Routing policy abstractions."""

from abc import ABC, abstractmethod
from typing import Any

from ai_platform.llm_gateway.routing.candidate_set import CandidateSet
from ai_platform.llm_gateway.routing.candidates import RoutingCandidate


class RoutingPolicy(ABC):
    """Base routing policy."""

    @abstractmethod
    def resolve_candidates(
        self,
        request: dict[str, Any],
    ) -> CandidateSet:
        """Resolve eligible routing candidates."""
        raise NotImplementedError


class ExplicitRoutingPolicy(RoutingPolicy):
    """Resolve routing candidates from explicit request information."""

    def __init__(
        self,
        default_candidates: list[RoutingCandidate] | None = None,
    ) -> None:
        self._default_candidates = list(default_candidates or [])

    def resolve_candidates(
        self,
        request: dict[str, Any],
    ) -> CandidateSet:
        """Resolve candidates from an explicit provider/model request."""

        provider = request.get("provider")
        model = request.get("model")

        if provider and model:
            return CandidateSet(
                [
                    RoutingCandidate(
                        provider=str(provider),
                        model=str(model),
                    )
                ]
            )

        if provider:
            candidates = [
                candidate
                for candidate in self._default_candidates
                if candidate.provider == provider
            ]

            return CandidateSet(candidates)

        if model:
            candidates = [
                candidate for candidate in self._default_candidates if candidate.model == model
            ]

            return CandidateSet(candidates)

        return CandidateSet(self._default_candidates)
