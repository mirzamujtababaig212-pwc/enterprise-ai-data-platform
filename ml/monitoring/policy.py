from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


DRIFT_POLICY_SCHEMA_VERSION = "v1"


class DriftStatus(StrEnum):
    STABLE = "stable"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True)
class DriftPolicy:
    """
    PSI thresholds used to classify production feature drift.
    """

    warning_psi: float = 0.10
    critical_psi: float = 0.25
    name: str | None = None

    def __post_init__(self) -> None:
        if self.name is not None and not self.name.strip():
            raise ValueError("name must not be empty")

        if self.warning_psi < 0.0:
            raise ValueError("warning_psi must be non-negative")

        if self.critical_psi < 0.0:
            raise ValueError("critical_psi must be non-negative")

        if self.warning_psi > self.critical_psi:
            raise ValueError("warning_psi must not exceed critical_psi")

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "warning_psi": self.warning_psi,
            "critical_psi": self.critical_psi,
        }


@dataclass(frozen=True)
class DriftDecision:
    """
    Result of evaluating feature PSI values against a DriftPolicy.
    """

    policy: DriftPolicy
    feature_status: dict[str, DriftStatus]
    overall_status: DriftStatus

    @property
    def stable(self) -> bool:
        return self.overall_status is DriftStatus.STABLE

    def as_dict(self) -> dict[str, object]:
        return {
            "drift_policy": self.policy.as_dict(),
            "feature_status": {
                feature_name: status.value for feature_name, status in self.feature_status.items()
            },
            "overall_status": self.overall_status.value,
        }


class DriftEvaluator:
    """
    Applies a DriftPolicy to feature PSI values.
    """

    @staticmethod
    def evaluate(
        feature_psi: dict[str, float],
        policy: DriftPolicy,
    ) -> DriftDecision:
        feature_status: dict[str, DriftStatus] = {}

        for feature_name, psi in feature_psi.items():
            if psi < 0.0:
                raise ValueError(f"PSI for feature {feature_name!r} must be non-negative")

            if psi >= policy.critical_psi:
                status = DriftStatus.CRITICAL
            elif psi >= policy.warning_psi:
                status = DriftStatus.WARNING
            else:
                status = DriftStatus.STABLE

            feature_status[feature_name] = status

        if any(status is DriftStatus.CRITICAL for status in feature_status.values()):
            overall_status = DriftStatus.CRITICAL
        elif any(status is DriftStatus.WARNING for status in feature_status.values()):
            overall_status = DriftStatus.WARNING
        else:
            overall_status = DriftStatus.STABLE

        return DriftDecision(
            policy=policy,
            feature_status=feature_status,
            overall_status=overall_status,
        )
