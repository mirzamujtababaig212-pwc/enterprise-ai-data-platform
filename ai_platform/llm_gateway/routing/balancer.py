"""Deterministic routing load balancing."""

from collections.abc import Sequence

from ai_platform.llm_gateway.routing.candidates import RoutingCandidate


class RoundRobinLoadBalancer:
    """Simple deterministic weighted round-robin balancer."""

    def __init__(self) -> None:
        self._index = 0

    def select(
        self,
        candidates: Sequence[RoutingCandidate],
    ) -> RoutingCandidate:
        """Select the next candidate."""

        if not candidates:
            raise ValueError("Cannot select from an empty candidate list.")

        weighted: list[RoutingCandidate] = []

        for candidate in candidates:
            weighted.extend([candidate] * candidate.weight)

        if not weighted:
            raise ValueError("No routable candidates are available.")

        candidate = weighted[self._index % len(weighted)]

        self._index += 1

        return candidate
