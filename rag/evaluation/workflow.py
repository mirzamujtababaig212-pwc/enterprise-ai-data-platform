from __future__ import annotations

from dataclasses import dataclass

from rag.evaluation.dataset import RetrievalEvaluationDataset
from rag.evaluation.evaluator import RetrievalEvaluator
from rag.evaluation.lineage import RetrievalEvaluationLineage
from rag.evaluation.models import RetrievalEvaluationResult
from rag.evaluation.policy import RetrievalEvaluationPolicy
from rag.evaluation.quality_gate import (
    RetrievalQualityGate,
    RetrievalQualityGateResult,
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
