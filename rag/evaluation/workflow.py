from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING


from rag.evaluation.dataset import RetrievalEvaluationDataset
from rag.evaluation.evaluator import RetrievalEvaluator
from rag.evaluation.lineage import RetrievalEvaluationLineage
from rag.evaluation.models import RetrievalEvaluationResult
from rag.evaluation.policy import RetrievalEvaluationPolicy
from rag.evaluation.run import RetrievalEvaluationRun
from rag.evaluation.quality_gate import (
    RetrievalQualityGate,
    RetrievalQualityGateResult,
)

if TYPE_CHECKING:
    from rag.evaluation.comparison.regression_policy import (
        RetrievalRegressionPolicy,
    )


@dataclass(frozen=True)
class RetrievalEvaluationWorkflowResult:
    """
    Combined result of retrieval evaluation and quality-gate enforcement.
    """

    dataset_name: str
    evaluation: RetrievalEvaluationResult
    quality_gate: RetrievalQualityGateResult
    lineage: RetrievalEvaluationLineage

    @property
    def passed(self) -> bool:
        return self.quality_gate.passed

    def as_dict(self) -> dict[str, object]:
        return {
            "dataset_name": self.dataset_name,
            "evaluation": self.evaluation.as_dict(),
            "quality_gate": self.quality_gate.as_dict(),
            "lineage": self.lineage.as_dict(),
        }

    def to_run(
        self,
        *,
        run_id: str,
        created_at: datetime,
        baseline: RetrievalEvaluationRun | None = None,
        regression_policy: RetrievalRegressionPolicy | None = None,
    ) -> RetrievalEvaluationRun:
        """
        Create an immutable evaluation-run artifact from this result.

        When both a baseline run and regression policy are supplied, the
        regression comparison is evaluated and attached to the returned run.
        The original workflow result and baseline remain unchanged.
        """
        if (baseline is None) != (regression_policy is None):
            raise ValueError("baseline and regression_policy must be provided together")

        run = RetrievalEvaluationRun(
            run_id=run_id,
            created_at=created_at,
            lineage=self.lineage,
            evaluation=self.evaluation,
            quality_gate=self.quality_gate,
        )

        if baseline is not None and regression_policy is not None:
            return run.with_regression(
                baseline=baseline,
                policy=regression_policy,
            )

        return run


class RetrievalEvaluationWorkflow:
    """
    Executes retrieval evaluation and applies a quality policy.
    """

    def __init__(
        self,
        evaluator: RetrievalEvaluator,
        policy: RetrievalEvaluationPolicy,
    ) -> None:
        self.evaluator = evaluator
        self.policy = policy

    async def run(
        self,
        dataset: RetrievalEvaluationDataset,
    ) -> RetrievalEvaluationWorkflowResult:
        evaluation = await self.evaluator.evaluate(dataset.cases)

        quality_gate = RetrievalQualityGate.evaluate(
            result=evaluation,
            policy=self.policy,
        )

        lineage = RetrievalEvaluationLineage(
            dataset_name=dataset.name,
            dataset_version=dataset.version,
            evaluation_policy_name=self.policy.name,
            min_recall_at_k=self.policy.min_recall_at_k,
            min_precision_at_k=self.policy.min_precision_at_k,
            min_mrr=self.policy.min_mrr,
            min_ndcg_at_k=self.policy.min_ndcg_at_k,
            max_mean_latency_ms=self.policy.max_mean_latency_ms,
            min_abstention_accuracy=self.policy.min_abstention_accuracy,
            evaluator_k=self.evaluator.k,
            min_relevance_score=self.evaluator.min_relevance_score,
            embedding_identity=self.evaluator.embedding_identity,
        )

        return RetrievalEvaluationWorkflowResult(
            dataset_name=dataset.name,
            evaluation=evaluation,
            quality_gate=quality_gate,
            lineage=lineage,
        )
