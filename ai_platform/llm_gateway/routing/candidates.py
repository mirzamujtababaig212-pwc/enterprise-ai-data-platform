"""Routing candidate models."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RoutingCandidate:
    """A provider/model combination eligible for routing."""

    provider: str
    model: str
    priority: int = 0
    weight: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.provider:
            raise ValueError("RoutingCandidate requires a provider.")

        if not self.model:
            raise ValueError("RoutingCandidate requires a model.")

        if self.weight < 1:
            raise ValueError("RoutingCandidate weight must be >= 1.")

    @property
    def key(self) -> str:
        """Return a stable provider/model identifier."""

        return f"{self.provider}:{self.model}"
