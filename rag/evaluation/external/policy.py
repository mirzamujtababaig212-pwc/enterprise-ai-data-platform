from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExternalEvaluationMetricPolicy:
    """
    Policy for one metric produced by an external evaluator.

    A metric can have a minimum threshold, a maximum threshold, or both.
    When required=True, missing evaluation evidence is a quality-gate failure.
    """

    provider: str
    evaluator: str
    metric_name: str
    minimum_value: float | None = None
    maximum_value: float | None = None
    required: bool = True

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ValueError("provider must not be empty")

        if not self.evaluator.strip():
            raise ValueError("evaluator must not be empty")

        if not self.metric_name.strip():
            raise ValueError("metric_name must not be empty")

        if self.minimum_value is None and self.maximum_value is None:
            raise ValueError("at least one of minimum_value or maximum_value must be configured")

        if (
            self.minimum_value is not None
            and self.maximum_value is not None
            and self.minimum_value > self.maximum_value
        ):
            raise ValueError("minimum_value must not exceed maximum_value")

    def as_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "evaluator": self.evaluator,
            "metric_name": self.metric_name,
            "minimum_value": self.minimum_value,
            "maximum_value": self.maximum_value,
            "required": self.required,
        }


@dataclass(frozen=True)
class ExternalEvaluationPolicy:
    """
    Collection of policies governing external evaluation evidence.
    """

    metrics: tuple[ExternalEvaluationMetricPolicy, ...]
    name: str | None = None

    def __post_init__(self) -> None:
        if self.name is not None and not self.name.strip():
            raise ValueError("name must not be empty")

        if not self.metrics:
            raise ValueError("at least one external evaluation metric policy is required")

        identities = [
            (metric.provider, metric.evaluator, metric.metric_name) for metric in self.metrics
        ]

        if len(identities) != len(set(identities)):
            raise ValueError("duplicate external evaluation metric policies are not allowed")

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "metrics": [metric.as_dict() for metric in self.metrics],
        }
