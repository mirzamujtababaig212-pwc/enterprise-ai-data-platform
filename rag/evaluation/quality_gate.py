from __future__ import annotations

from dataclasses import dataclass

from .models import RetrievalEvaluationResult
from .policy import RetrievalEvaluationPolicy


@dataclass(frozen=True)
class RetrievalQualityGateResult:
    """
    Result of evaluating retrieval against a RetrievalEvaluationPolicy.
    """

    passed: bool
    errors: tuple[str, ...]
    metrics: dict[str, float]
    policy: RetrievalEvaluationPolicy

    def as_dict(self) -> dict[str, object]:
        return {
            "quality_gate_passed": self.passed,
            "quality_gate_policy": self.policy.name,
            "quality_gate_errors": self.errors,
            "quality_gate_metrics": self.metrics,
        }


class RetrievalQualityGate:
    """
    Applies a RetrievalEvaluationPolicy to a RetrievalEvaluationResult.
    """

    @staticmethod
    def evaluate(
        result: RetrievalEvaluationResult,
        policy: RetrievalEvaluationPolicy,
    ) -> RetrievalQualityGateResult:
        errors: list[str] = []

        metrics = result.as_dict()

        checks = (
            (
                "retrieval_recall_at_k",
                policy.min_recall_at_k,
                result.recall_at_k,
            ),
            (
                "retrieval_precision_at_k",
                policy.min_precision_at_k,
                result.precision_at_k,
            ),
            (
                "retrieval_mrr",
                policy.min_mrr,
                result.mrr,
            ),
            (
                "retrieval_ndcg_at_k",
                policy.min_ndcg_at_k,
                result.ndcg_at_k,
            ),
        )

        for metric_name, minimum, actual in checks:
            if minimum is None:
                continue

            if actual is None:
                errors.append(
                    f"{metric_name} is unavailable but minimum " f"{minimum:.4f} is required"
                )
                continue

            if actual < minimum:
                errors.append(
                    f"{metric_name}={actual:.4f} is below " f"required minimum {minimum:.4f}"
                )

        if policy.max_mean_latency_ms is not None:
            if result.mean_latency_ms > policy.max_mean_latency_ms:
                errors.append(
                    f"retrieval_mean_latency_ms="
                    f"{result.mean_latency_ms:.2f} exceeds maximum "
                    f"{policy.max_mean_latency_ms:.2f}"
                )

        return RetrievalQualityGateResult(
            passed=not errors,
            errors=tuple(errors),
            metrics=metrics,
            policy=policy,
        )
