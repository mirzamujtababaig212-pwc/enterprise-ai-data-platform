from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

from .models import RetrievalEvaluationCase


@dataclass(frozen=True)
class RetrievalEvaluationDataset:
    """
    Immutable collection of labeled retrieval evaluation cases.
    """

    name: str
    version: str
    cases: tuple[RetrievalEvaluationCase, ...]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name must not be empty")

        if not self.version.strip():
            raise ValueError("version must not be empty")

        if not self.cases:
            raise ValueError("cases must not be empty")

    @classmethod
    def from_cases(
        cls,
        name: str,
        cases: Sequence[RetrievalEvaluationCase],
        *,
        version: str = "unversioned",
    ) -> RetrievalEvaluationDataset:
        return cls(
            name=name,
            version=version,
            cases=tuple(cases),
        )

    @property
    def size(self) -> int:
        return len(self.cases)
