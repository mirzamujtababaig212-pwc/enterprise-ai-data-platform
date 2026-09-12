from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy.orm import Session

from rag.evaluation.comparison.regression_policy import RetrievalRegressionPolicy
from rag.evaluation.composite_release import CompositeEvaluationReleaseDecision
from rag.evaluation.dataset_registry import EvaluationDatasetRegistry
from rag.evaluation.evaluator import RetrievalEvaluator
from rag.evaluation.policy import RetrievalEvaluationPolicy
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
from rag.evaluation.workflow import RetrievalEvaluationWorkflow
from typing import Iterable


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
        dataset_name: str,
        dataset_version: str,
        run_id: str,
        created_at: datetime,
        evaluation_policy: RetrievalEvaluationPolicy,
        external_release_policy: ExternalEvaluationReleasePolicy,
        k: int = 5,
        min_relevance_score: float | None = None,
        baseline: RetrievalEvaluationRun | None = None,
        baseline_run_id: str | None = None,
        baselines: Iterable[RetrievalEvaluationRun] | None = None,
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

        definition = EvaluationDatasetRegistry.get(
            name=dataset_name,
            version=dataset_version,
        )

        dataset = definition.build_dataset()

        if dataset.name != dataset_name or dataset.version != dataset_version:
            raise ValueError(
                "evaluation dataset definition does not match requested "
                f"dataset: requested={dataset_name!r}/{dataset_version!r}, "
                f"resolved={dataset.name!r}/{dataset.version!r}"
            )

        retriever = await definition.build_retriever()

        evaluator = RetrievalEvaluator(
            retriever=retriever,
            k=k,
            min_relevance_score=min_relevance_score,
            embedding_identity=definition.build_embedding_identity(),
        )

        workflow = RetrievalEvaluationWorkflow(
            evaluator=evaluator,
            policy=evaluation_policy,
            retrieval_artifact=definition.build_retrieval_artifact(),
        )

        evaluation_result = await workflow.run(dataset)

        workflow_result = CompositeEvaluationWorkflow.run(
            result=evaluation_result,
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
