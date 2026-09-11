from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalRegressionPolicy:
    """
    Tolerance policy for deciding whether evaluation metric changes
    constitute meaningful regressions.

    Thresholds represent the maximum tolerated degradation.

    Higher-is-better metrics:
        degradation = baseline - candidate

    Lower-is-better metrics:
        degradation = candidate - baseline
    """

    max_recall_at_k_degradation: float = 0.0
    max_precision_at_k_degradation: float = 0.0
    max_mrr_degradation: float = 0.0
    max_ndcg_at_k_degradation: float = 0.0
    max_mean_latency_ms_increase: float = 0.0
    max_abstention_accuracy_degradation: float = 0.0
    name: str | None = None

    def __post_init__(self) -> None:
        thresholds = {
            "max_recall_at_k_degradation": self.max_recall_at_k_degradation,
            "max_precision_at_k_degradation": self.max_precision_at_k_degradation,
            "max_mrr_degradation": self.max_mrr_degradation,
            "max_ndcg_at_k_degradation": self.max_ndcg_at_k_degradation,
            "max_mean_latency_ms_increase": self.max_mean_latency_ms_increase,
            "max_abstention_accuracy_degradation": (self.max_abstention_accuracy_degradation),
        }

        for field_name, value in thresholds.items():
            if value < 0.0:
                raise ValueError(f"{field_name} must be >= 0")

        if self.name is not None and not self.name.strip():
            raise ValueError("name must not be empty")

    def tolerance_for(self, metric_name: str) -> float:
        tolerances = {
            "recall_at_k": self.max_recall_at_k_degradation,
            "precision_at_k": self.max_precision_at_k_degradation,
            "mrr": self.max_mrr_degradation,
            "ndcg_at_k": self.max_ndcg_at_k_degradation,
            "mean_latency_ms": self.max_mean_latency_ms_increase,
            "abstention_accuracy": self.max_abstention_accuracy_degradation,
        }

        try:
            return tolerances[metric_name]
        except KeyError as exc:
            raise ValueError(f"Unsupported evaluation metric: {metric_name!r}") from exc

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "max_recall_at_k_degradation": self.max_recall_at_k_degradation,
            "max_precision_at_k_degradation": (self.max_precision_at_k_degradation),
            "max_mrr_degradation": self.max_mrr_degradation,
            "max_ndcg_at_k_degradation": self.max_ndcg_at_k_degradation,
            "max_mean_latency_ms_increase": (self.max_mean_latency_ms_increase),
            "max_abstention_accuracy_degradation": (self.max_abstention_accuracy_degradation),
        }


@dataclass(frozen=True)
class RetrievalRegressionPolicyResult:
    """Result of applying a regression policy to a run comparison."""

    passed: bool
    errors: tuple[str, ...]
    policy_name: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "errors": list(self.errors),
            "policy_name": self.policy_name,
        }


class RetrievalRegressionPolicyEvaluator:
    """Apply regression tolerances to a run comparison."""

    _HIGHER_IS_BETTER = {
        "recall_at_k",
        "precision_at_k",
        "mrr",
        "ndcg_at_k",
        "abstention_accuracy",
    }

    _LOWER_IS_BETTER = {"mean_latency_ms"}

    @classmethod
    def evaluate(
        cls,
        comparison,
        policy: RetrievalRegressionPolicy,
    ) -> RetrievalRegressionPolicyResult:
        errors: list[str] = []

        for metric_name, metric_comparison in comparison.metrics.items():
            tolerance = policy.tolerance_for(metric_name)

            if metric_name in cls._HIGHER_IS_BETTER:
                degradation = max(
                    0.0,
                    metric_comparison.baseline - metric_comparison.candidate,
                )
            elif metric_name in cls._LOWER_IS_BETTER:
                degradation = max(
                    0.0,
                    metric_comparison.candidate - metric_comparison.baseline,
                )
            else:
                raise ValueError(f"Unsupported evaluation metric: {metric_name!r}")

            if degradation > tolerance:
                errors.append(
                    f"{metric_name} degradation exceeds tolerance: "
                    f"degradation={degradation}, "
                    f"tolerance={tolerance}."
                )

        return RetrievalRegressionPolicyResult(
            passed=not errors,
            errors=tuple(errors),
            policy_name=policy.name,
        )
