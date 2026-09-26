from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .policy import DriftDecision, DriftStatus


DRIFT_ACTION_POLICY_SCHEMA_VERSION = "v1"


class DriftAction(StrEnum):
    NO_ACTION = "no_action"
    INVESTIGATE = "investigate"
    RETRAIN_REVIEW = "retrain_review"


@dataclass(frozen=True)
class DriftActionDecision:
    """
    Operational action derived from a DriftDecision.
    """

    decision: DriftDecision
    action: DriftAction

    @property
    def requires_action(self) -> bool:
        return self.action is not DriftAction.NO_ACTION

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": DRIFT_ACTION_POLICY_SCHEMA_VERSION,
            "action": self.action.value,
            "requires_action": self.requires_action,
            "drift_decision": self.decision.as_dict(),
        }


class DriftActionPolicy:
    """
    Maps drift severity to an operational monitoring action.

    This policy classifies the required next step only. It does not
    execute investigation, retraining, deployment, or notification.
    """

    @staticmethod
    def evaluate(decision: DriftDecision) -> DriftActionDecision:
        actions = {
            DriftStatus.STABLE: DriftAction.NO_ACTION,
            DriftStatus.WARNING: DriftAction.INVESTIGATE,
            DriftStatus.CRITICAL: DriftAction.RETRAIN_REVIEW,
        }

        return DriftActionDecision(
            decision=decision,
            action=actions[decision.overall_status],
        )
