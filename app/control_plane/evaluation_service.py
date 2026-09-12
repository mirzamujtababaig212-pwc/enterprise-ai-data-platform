from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy.orm import Session

from rag.evaluation.comparison.regression_policy import RetrievalRegressionPolicy
from rag.evaluation.composite_release import CompositeEvaluationReleaseDecision
from rag.evaluation.composite_workflow import CompositeEvaluationWorkflow
from rag.evaluation.external.models import ExternalEvaluationResult
from rag.evaluation.external.policy import ExternalEvaluationPolicy
from rag.evaluation.external.release import ExternalEvaluationReleasePolicy
from rag.evaluation.run import RetrievalEvaluationRun
from rag.evaluation.run_store import DuplicateEvaluationRunError
from rag.evaluation.stores.postgres import (
    PostgreSQLRetrievalEvaluationRunStore,
)
from rag.evaluation.stores.release_decision import (
    PostgreSQLRetrievalEvaluationReleaseDecisionStore,
)
from rag.evaluation.workflow import RetrievalEvaluationWorkflowResult


@dataclass(frozen=True)
class EvaluationExecutionResult:
    """Persisted evaluation run and its composite release decision."""

    run: RetrievalEvaluationRun
    release_decision: CompositeEvaluationReleaseDecision


class EvaluationExecutionService:
    """
    Application-level orchestration for retrieval evaluation execution.

    The domain evaluation workflow remains persistence-free. This service owns
    the application transaction boundary and persists the evaluation run and
    its derived composite release decision atomically.
    """

    def __init__(
        self,
        *,
        session: Session,
        run_store: PostgreSQLRetrievalEvaluationRunStore | None = None,
        release_decision_store: PostgreSQLRetrievalEvaluationReleaseDecisionStore | None = None,
    ) -> None:
        self._session = session
        self._run_store = run_store or PostgreSQLRetrievalEvaluationRunStore(session)
        self._release_decision_store = (
            release_decision_store or PostgreSQLRetrievalEvaluationReleaseDecisionStore(session)
        )

    async def execute(
        self,
        *,
        result: RetrievalEvaluationWorkflowResult,
        run_id: str,
        created_at: datetime,
        external_release_policy: ExternalEvaluationReleasePolicy,
        baseline=None,
        baseline_run_id: str | None = None,
        baselines=None,
        regression_policy: RetrievalRegressionPolicy | None = None,
        external_evaluations: tuple[ExternalEvaluationResult, ...] = (),
        external_policy: ExternalEvaluationPolicy | None = None,
    ) -> EvaluationExecutionResult:
        """
        Build, release-evaluate, and atomically persist an evaluation run.

        Both persistence operations participate in the same SQLAlchemy
        transaction. Existing PostgreSQL store callers retain their default
        commit=True behavior; this service explicitly suppresses those
        individual commits so the two artifacts commit together.
        """
        existing_run = await self._run_store.get(run_id)

        if existing_run is not None:
            raise DuplicateEvaluationRunError(f"evaluation run already exists: {run_id}")

        workflow_result = CompositeEvaluationWorkflow.run(
            result=result,
            run_id=run_id,
            created_at=created_at,
            external_release_policy=external_release_policy,
            baseline=baseline,
            baseline_run_id=baseline_run_id,
            baselines=baselines,
            regression_policy=regression_policy,
            external_evaluations=external_evaluations,
            external_policy=external_policy,
        )

        try:
            await self._run_store.save(
                workflow_result.run,
                commit=False,
            )

            await self._release_decision_store.save(
                workflow_result.release_decision,
                commit=False,
            )

            self._session.commit()

        except Exception:
            self._session.rollback()
            raise

        return EvaluationExecutionResult(
            run=workflow_result.run,
            release_decision=workflow_result.release_decision,
        )
