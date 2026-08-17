"""Candidate collection utilities for routing."""

from collections.abc import Iterable, Iterator

from ai_platform.llm_gateway.routing.candidates import RoutingCandidate


class CandidateSet:
    """Ordered collection of routing candidates."""

    def __init__(
        self,
        candidates: Iterable[RoutingCandidate] | None = None,
    ) -> None:
        self._candidates = list(candidates or [])

    def __iter__(self) -> Iterator[RoutingCandidate]:
        return iter(self._candidates)

    def __len__(self) -> int:
        return len(self._candidates)

    def __bool__(self) -> bool:
        return bool(self._candidates)

    def add(self, candidate: RoutingCandidate) -> None:
        """Add a routing candidate."""

        self._candidates.append(candidate)

    def as_list(self) -> list[RoutingCandidate]:
        """Return a copy of the candidate list."""

        return list(self._candidates)

    def sorted_by_priority(self) -> list[RoutingCandidate]:
        """Return candidates ordered by descending priority."""

        return sorted(
            self._candidates,
            key=lambda candidate: candidate.priority,
            reverse=True,
        )

    def providers(self) -> list[str]:
        """Return unique providers preserving order."""

        result: list[str] = []

        for candidate in self._candidates:
            if candidate.provider not in result:
                result.append(candidate.provider)

        return result

    def models(self) -> list[str]:
        """Return unique models preserving order."""

        result: list[str] = []

        for candidate in self._candidates:
            if candidate.model not in result:
                result.append(candidate.model)

        return result
