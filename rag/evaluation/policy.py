from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalEvaluationPolicy:
    """
    Quality thresholds required for retrieval to pass evaluation.
    """

    min_recall_at_k: float | None = None
    min_precision_at_k: float | None = None
    min_mrr: float | None = None
    min_ndcg_at_k: float | None = None
    max_mean_latency_ms: float | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        if self.name is not None and not self.name.strip():
            raise ValueError("name must not be empty")

        quality_thresholds = {
            "min_recall_at_k": self.min_recall_at_k,
            "min_precision_at_k": self.min_precision_at_k,
            "min_mrr": self.min_mrr,
            "min_ndcg_at_k": self.min_ndcg_at_k,
        }

        for name, value in quality_thresholds.items():
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0.0 and 1.0")

        if self.max_mean_latency_ms is not None and self.max_mean_latency_ms < 0.0:
            raise ValueError("max_mean_latency_ms must be non-negative")

        if all(
            value is None
            for value in (
                self.min_recall_at_k,
                self.min_precision_at_k,
                self.min_mrr,
                self.min_ndcg_at_k,
                self.max_mean_latency_ms,
            )
        ):
            raise ValueError("at least one evaluation threshold must be configured")

    def as_dict(self) -> dict[str, float]:
        thresholds = {
            "min_recall_at_k": self.min_recall_at_k,
            "min_precision_at_k": self.min_precision_at_k,
            "min_mrr": self.min_mrr,
            "min_ndcg_at_k": self.min_ndcg_at_k,
            "max_mean_latency_ms": self.max_mean_latency_ms,
        }
        return {key: value for key, value in thresholds.items() if value is not None}
