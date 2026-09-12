from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class ExternalEvaluationRequest:
    provider: str
    evaluator: str

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ValueError("provider must be non-empty")
        if not self.evaluator.strip():
            raise ValueError("evaluator must be non-empty")


@dataclass(frozen=True)
class ExternalEvaluationSample:
    """Provider-neutral sample for external evaluation frameworks."""

    query: str
    retrieved_contexts: tuple[str, ...]
    response: str
    reference: str | None = None

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("query must not be empty")

        if not self.retrieved_contexts:
            raise ValueError("retrieved_contexts must not be empty")

        if any(not context.strip() for context in self.retrieved_contexts):
            raise ValueError("retrieved_contexts must not contain empty values")

        if not self.response.strip():
            raise ValueError("response must not be empty")

        if self.reference is not None and not self.reference.strip():
            raise ValueError("reference must not be empty when provided")


@dataclass(frozen=True)
class ExternalEvaluationResult:
    """Aggregate result produced by an external evaluation provider."""

    provider: str
    evaluator: str
    metrics: Mapping[str, float]
    evaluated_samples: int
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ValueError("provider must not be empty")

        if not self.evaluator.strip():
            raise ValueError("evaluator must not be empty")

        if self.evaluated_samples < 0:
            raise ValueError("evaluated_samples must be >= 0")

        for name, value in self.metrics.items():
            if not name.strip():
                raise ValueError("metric names must not be empty")

            if not isinstance(value, (int, float)):
                raise TypeError(f"metric {name!r} must be numeric, got {type(value).__name__}")

    def as_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "evaluator": self.evaluator,
            "metrics": dict(self.metrics),
            "evaluated_samples": self.evaluated_samples,
            "metadata": dict(self.metadata or {}),
        }
