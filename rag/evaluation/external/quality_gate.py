from __future__ import annotations

from dataclasses import dataclass

from .models import ExternalEvaluationResult
from .policy import ExternalEvaluationMetricPolicy, ExternalEvaluationPolicy


@dataclass(frozen=True)
class ExternalEvaluationQualityGateResult:
    """
    Result of evaluating external evidence against an external policy.
    """

    passed: bool
    errors: tuple[str, ...]
    metrics: dict[str, float]
    policy: ExternalEvaluationPolicy

    def as_dict(self) -> dict[str, object]:
        return {
            "quality_gate_passed": self.passed,
            "quality_gate_policy": self.policy.name,
            "quality_gate_errors": self.errors,
            "quality_gate_metrics": self.metrics,
        }


class ExternalEvaluationQualityGate:
    """
    Applies an ExternalEvaluationPolicy to external evaluation evidence.
    """

    @staticmethod
    def evaluate(
        results: tuple[ExternalEvaluationResult, ...],
        policy: ExternalEvaluationPolicy,
    ) -> ExternalEvaluationQualityGateResult:
        errors: list[str] = []
        metrics: dict[str, float] = {}

        for result in results:
            for metric_name, value in result.metrics.items():
                key = f"{result.provider}/" f"{result.evaluator}/" f"{metric_name}"
                metrics[key] = float(value)

        for metric_policy in policy.metrics:
            matching_result = ExternalEvaluationQualityGate._find_result(
                results,
                metric_policy,
            )

            if matching_result is None:
                if metric_policy.required:
                    errors.append(
                        "required external evaluation evidence is missing: "
                        f"provider={metric_policy.provider!r}, "
                        f"evaluator={metric_policy.evaluator!r}, "
                        f"metric={metric_policy.metric_name!r}"
                    )
                continue

            actual = matching_result.metrics.get(metric_policy.metric_name)

            if actual is None:
                if metric_policy.required:
                    errors.append(
                        "required external evaluation metric is missing: "
                        f"provider={metric_policy.provider!r}, "
                        f"evaluator={metric_policy.evaluator!r}, "
                        f"metric={metric_policy.metric_name!r}"
                    )
                continue

            if metric_policy.minimum_value is not None and actual < metric_policy.minimum_value:
                errors.append(
                    f"external_{metric_policy.metric_name}={actual:.4f} "
                    f"is below required minimum "
                    f"{metric_policy.minimum_value:.4f}"
                )

            if metric_policy.maximum_value is not None and actual > metric_policy.maximum_value:
                errors.append(
                    f"external_{metric_policy.metric_name}={actual:.4f} "
                    f"exceeds maximum "
                    f"{metric_policy.maximum_value:.4f}"
                )

        return ExternalEvaluationQualityGateResult(
            passed=not errors,
            errors=tuple(errors),
            metrics=metrics,
            policy=policy,
        )

    @staticmethod
    def _find_result(
        results: tuple[ExternalEvaluationResult, ...],
        policy: ExternalEvaluationMetricPolicy,
    ) -> ExternalEvaluationResult | None:
        for result in results:
            if result.provider == policy.provider and result.evaluator == policy.evaluator:
                return result

        return None
