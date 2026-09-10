from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GovernancePolicy:
    """
    Deterministic metadata requirements for governed retrieval.

    Each key/value pair must match the corresponding DocumentChunk
    metadata exactly.
    """

    required_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for key in self.required_metadata:
            if not isinstance(key, str) or not key.strip():
                raise ValueError("Governance metadata keys must be non-empty strings.")

    def to_metadata_filter(self) -> dict[str, Any]:
        """
        Convert this policy into the existing VectorStore metadata filter.
        """

        return dict(self.required_metadata)
