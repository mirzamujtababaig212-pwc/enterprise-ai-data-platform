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
    ) -> None:
        self._execution_service = execution_service

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
    ) -> EvaluationApplicationResult:
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
