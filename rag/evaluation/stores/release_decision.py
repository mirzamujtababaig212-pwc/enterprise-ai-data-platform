from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.control_plane.persistence.models import (
    RetrievalEvaluationReleaseDecisionRecord,
)
from rag.evaluation.composite_release import CompositeEvaluationReleaseDecision
from rag.evaluation.external.release import (
    ExternalEvaluationReleaseDecision,
    ExternalEvaluationReleasePolicy,
)
from rag.evaluation.release import RetrievalEvaluationReleaseDecision
from rag.evaluation.release_decision_store import (
    RetrievalEvaluationReleaseDecisionStore,
)


class PostgreSQLRetrievalEvaluationReleaseDecisionStore(RetrievalEvaluationReleaseDecisionStore):
    """
    Durable PostgreSQL persistence for composite retrieval release decisions.

    The decision is persisted as a derived audit artifact. It is intentionally
    separate from RetrievalEvaluationRun so evaluation evidence and release
    policy decisions remain independently versionable.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    async def save(
        self,
        decision: CompositeEvaluationReleaseDecision,
    ) -> None:
        try:
            payload = _serialize_decision(decision)

            record = self._session.scalar(
                select(RetrievalEvaluationReleaseDecisionRecord).where(
                    RetrievalEvaluationReleaseDecisionRecord.run_id == decision.run_id
                )
            )

            if record is None:
                self._session.add(RetrievalEvaluationReleaseDecisionRecord(**payload))
            else:
                record.passed = payload["passed"]
                record.errors = payload["errors"]
                record.native = payload["native"]
                record.external = payload["external"]

            self._session.commit()

        except Exception:
            self._session.rollback()
            raise

    async def get(
        self,
        run_id: str,
    ) -> CompositeEvaluationReleaseDecision | None:
        record = self._session.scalar(
            select(RetrievalEvaluationReleaseDecisionRecord).where(
                RetrievalEvaluationReleaseDecisionRecord.run_id == run_id
            )
        )

        if record is None:
            return None

        return _deserialize_decision(record)


def _serialize_decision(
    decision: CompositeEvaluationReleaseDecision,
) -> dict[str, object]:
    return {
        "run_id": decision.run_id,
        "passed": decision.passed,
        "errors": list(decision.errors),
        "native": {
            "run_id": decision.native.run_id,
            "passed": decision.native.passed,
            "errors": list(decision.native.errors),
        },
        "external": {
            "passed": decision.external.passed,
            "errors": list(decision.external.errors),
            "policy": decision.external.policy.as_dict(),
        },
    }


def _deserialize_decision(
    record: RetrievalEvaluationReleaseDecisionRecord,
) -> CompositeEvaluationReleaseDecision:
    native_data = dict(record.native)
    external_data = dict(record.external)

    native = RetrievalEvaluationReleaseDecision(
        run_id=native_data["run_id"],
        passed=native_data["passed"],
        errors=tuple(native_data["errors"]),
    )

    external_policy_data = dict(external_data["policy"])
    external_policy = ExternalEvaluationReleasePolicy(
        name=external_policy_data["name"],
        required=external_policy_data["required"],
    )

    external = ExternalEvaluationReleaseDecision(
        passed=external_data["passed"],
        errors=tuple(external_data["errors"]),
        policy=external_policy,
    )

    return CompositeEvaluationReleaseDecision(
        run_id=record.run_id,
        native=native,
        external=external,
        passed=record.passed,
        errors=tuple(record.errors),
    )
