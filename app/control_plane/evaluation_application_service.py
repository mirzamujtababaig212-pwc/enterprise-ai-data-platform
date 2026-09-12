from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from rag.evaluation.comparison.regression_policy import RetrievalRegressionPolicy
from rag.evaluation.dataset_registry import EvaluationDatasetRegistry
from rag.evaluation.external.models import ExternalEvaluationResult
from rag.evaluation.external.policy import ExternalEvaluationPolicy
from rag.evaluation.external.release import ExternalEvaluationReleasePolicy
from rag.evaluation.evaluator import RetrievalEvaluator
from rag.evaluation.policy import RetrievalEvaluationPolicy
from rag.evaluation.run import RetrievalEvaluationRun
from rag.evaluation.workflow import RetrievalEvaluationWorkflow

from app.control_plane.evaluation_service import (
    EvaluationExecutionResult,
    EvaluationExecutionService,
)
from rag.evaluation.run_store import RetrievalEvaluationRunStore


class EvaluationBaselineNotFoundError(ValueError):
    """Raised when an explicitly requested evaluation baseline is missing."""


class EvaluationRegressionConfigurationError(ValueError):
    """Raised when baseline and regression configuration are inconsistent."""


@dataclass(frozen=True)
class EvaluationApplicationResult:
    """Application-level evaluation execution result."""

    execution: EvaluationExecutionResult

    @property
    def run(self) -> RetrievalEvaluationRun:
        return self.execution.run

    @property
    def release_decision(self):
        return self.execution.release_decision


class EvaluationApplicationService:
    """
    Application orchestration for governed retrieval evaluation.

    Dataset resolution is explicit and versioned. Evaluation uses an isolated
    dataset-specific retrieval runtime and delegates persistence/transaction
    handling to EvaluationExecutionService.
    """

    def __init__(
        self,
        *,
        execution_service: EvaluationExecutionService,
        run_store: RetrievalEvaluationRunStore,
    ) -> None:
        self._execution_service = execution_service
        self._run_store = run_store

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
        baseline_run_id: str | None = None,
        baselines: Iterable[RetrievalEvaluationRun] | None = None,
        regression_policy: RetrievalRegressionPolicy | None = None,
        external_evaluations: tuple[ExternalEvaluationResult, ...] = (),
        external_policy: ExternalEvaluationPolicy | None = None,
    ) -> EvaluationApplicationResult:
        if baseline_run_id is not None and regression_policy is None:
            raise EvaluationRegressionConfigurationError(
                "regression_policy must be provided when baseline_run_id is provided"
            )

        if baseline_run_id is None and regression_policy is not None:
            raise EvaluationRegressionConfigurationError(
                "baseline_run_id must be provided when regression_policy is provided"
            )

        baseline = None
        if baseline_run_id is not None:
            baseline = await self._run_store.get(baseline_run_id)
            if baseline is None:
                raise EvaluationBaselineNotFoundError(
                    f"baseline evaluation run not found: {baseline_run_id}"
                )

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
        )

        workflow_result = await workflow.run(dataset)

        execution = await self._execution_service.execute(
            result=workflow_result,
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

        return EvaluationApplicationResult(execution=execution)
