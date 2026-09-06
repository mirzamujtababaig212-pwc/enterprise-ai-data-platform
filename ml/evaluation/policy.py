from __future__ import annotations

from dataclasses import dataclass

from .evaluator import EvaluationResult


@dataclass(frozen=True)
class EvaluationPolicy:
    """
    Minimum quality thresholds required for a model to pass evaluation.
    """

    min_accuracy: float | None = None
    min_precision: float | None = None
    min_recall: float | None = None
    min_f1: float | None = None
    min_roc_auc: float | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        if self.name is not None and not self.name.strip():
            raise ValueError("name must not be empty")

        thresholds = {
            "min_accuracy": self.min_accuracy,
            "min_precision": self.min_precision,
            "min_recall": self.min_recall,
            "min_f1": self.min_f1,
            "min_roc_auc": self.min_roc_auc,
        }

        for name, value in thresholds.items():
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0.0 and 1.0")

    def as_dict(self) -> dict[str, float]:
        thresholds = {
            "min_accuracy": self.min_accuracy,
            "min_precision": self.min_precision,
            "min_recall": self.min_recall,
            "min_f1": self.min_f1,
            "min_roc_auc": self.min_roc_auc,
        }
        return {key: value for key, value in thresholds.items() if value is not None}


@dataclass(frozen=True)
class QualityGateResult:
    """
    Result of evaluating a model against an EvaluationPolicy.
    """

    passed: bool
    errors: tuple[str, ...]
    metrics: dict[str, float]
    policy: EvaluationPolicy

    def as_dict(self) -> dict[str, object]:
        return {
            "quality_gate_passed": self.passed,
            "quality_gate_policy": self.policy.name,
            "quality_gate_errors": self.errors,
            "quality_gate_metrics": self.metrics,
        }


class EvaluationQualityGate:
    """
    Applies an EvaluationPolicy to an EvaluationResult.
    """

    @staticmethod
    def evaluate(
        result: EvaluationResult,
        policy: EvaluationPolicy,
    ) -> QualityGateResult:
        errors: list[str] = []

        metrics = result.as_dict()

        checks = (
            ("validation_accuracy", policy.min_accuracy, result.accuracy),
            ("validation_precision", policy.min_precision, result.precision),
            ("validation_recall", policy.min_recall, result.recall),
            ("validation_f1", policy.min_f1, result.f1),
            ("validation_roc_auc", policy.min_roc_auc, result.roc_auc),
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

        return QualityGateResult(
            passed=not errors,
            errors=tuple(errors),
            metrics=metrics,
            policy=policy,
        )
