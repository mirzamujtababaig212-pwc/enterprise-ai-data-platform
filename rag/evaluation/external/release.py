from __future__ import annotations

from dataclasses import dataclass

from .quality_gate import ExternalEvaluationQualityGateResult


@dataclass(frozen=True)
class ExternalEvaluationReleasePolicy:
    """
    Policy governing whether external evaluation evidence is required for
    release.

    Metric thresholds remain owned by ExternalEvaluationPolicy and its
    quality gate. This policy only determines whether external evidence is
    mandatory at release time.
    """

    name: str | None = None
    required: bool = False

    def __post_init__(self) -> None:
        if self.name is not None and not self.name.strip():
            raise ValueError("name must not be empty")

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "required": self.required,
        }


@dataclass(frozen=True)
class ExternalEvaluationReleaseDecision:
    """
    Immutable release decision derived from external evaluation evidence.
    """

    passed: bool
    errors: tuple[str, ...]
    policy: ExternalEvaluationReleasePolicy

    def as_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "errors": self.errors,
            "policy": self.policy.as_dict(),
        }


class ExternalEvaluationReleaseGate:
    """
    Evaluate whether external evaluation state is acceptable for release.

    This gate does not modify the native retrieval release decision.
    """

    @staticmethod
    def evaluate(
        *,
        quality_gate: ExternalEvaluationQualityGateResult | None,
        policy: ExternalEvaluationReleasePolicy,
    ) -> ExternalEvaluationReleaseDecision:
        errors: list[str] = []

        if quality_gate is None:
            if policy.required:
                errors.append("required external evaluation evidence is missing")
        elif not quality_gate.passed:
            errors.extend(quality_gate.errors)

        return ExternalEvaluationReleaseDecision(
            passed=not errors,
            errors=tuple(errors),
            policy=policy,
        )
