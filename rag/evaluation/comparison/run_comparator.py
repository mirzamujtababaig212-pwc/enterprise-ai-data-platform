from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from rag.evaluation.run import RetrievalEvaluationRun


class RetrievalEvaluationMetricStatus(str, Enum):
    """Direction of change for an evaluation metric."""

    IMPROVED = "improved"
    UNCHANGED = "unchanged"
    REGRESSED = "regressed"


@dataclass(frozen=True)
class RetrievalEvaluationMetricComparison:
    """Comparison of one retrieval evaluation metric."""

    baseline: float
    candidate: float
    delta: float
    status: RetrievalEvaluationMetricStatus


@dataclass(frozen=True)
class RetrievalEvaluationRunComparison:
    """Immutable comparison between two compatible evaluation runs."""

    baseline_run_id: str
    candidate_run_id: str
    metrics: dict[str, RetrievalEvaluationMetricComparison]

    @property
    def regressed(self) -> bool:
        return any(
            comparison.status is RetrievalEvaluationMetricStatus.REGRESSED
            for comparison in self.metrics.values()
        )

    @property
    def passed(self) -> bool:
        return not self.regressed

    def as_dict(self) -> dict[str, object]:
        return {
            "baseline_run_id": self.baseline_run_id,
            "candidate_run_id": self.candidate_run_id,
            "regressed": self.regressed,
            "passed": self.passed,
            "metrics": {
                name: {
                    "baseline": comparison.baseline,
                    "candidate": comparison.candidate,
                    "delta": comparison.delta,
                    "status": comparison.status.value,
                }
                for name, comparison in self.metrics.items()
            },
        }


class RetrievalEvaluationRunComparator:
    """Compare metrics from two compatible retrieval evaluation runs."""

    _HIGHER_IS_BETTER = (
        "recall_at_k",
        "precision_at_k",
        "mrr",
        "ndcg_at_k",
        "abstention_accuracy",
    )

    _LOWER_IS_BETTER = ("mean_latency_ms",)

    @classmethod
    def compare(
        cls,
        baseline: RetrievalEvaluationRun,
        candidate: RetrievalEvaluationRun,
    ) -> RetrievalEvaluationRunComparison:
        cls._validate_compatibility(baseline, candidate)

        metrics = {}

        for metric_name in cls._HIGHER_IS_BETTER:
            baseline_value = cls._metric_value(baseline, metric_name)
            candidate_value = cls._metric_value(candidate, metric_name)

            metrics[metric_name] = cls._compare_metric(
                baseline_value,
                candidate_value,
                higher_is_better=True,
            )

        for metric_name in cls._LOWER_IS_BETTER:
            baseline_value = cls._metric_value(baseline, metric_name)
            candidate_value = cls._metric_value(candidate, metric_name)

            metrics[metric_name] = cls._compare_metric(
                baseline_value,
                candidate_value,
                higher_is_better=False,
            )

        return RetrievalEvaluationRunComparison(
            baseline_run_id=baseline.run_id,
            candidate_run_id=candidate.run_id,
            metrics=metrics,
        )

    @staticmethod
    def _metric_value(
        run: RetrievalEvaluationRun,
        metric_name: str,
    ) -> float:
        evaluation = run.evaluation

        if metric_name == "recall_at_k":
            return evaluation.recall_at_k
        if metric_name == "precision_at_k":
            return evaluation.precision_at_k
        if metric_name == "mrr":
            return evaluation.mrr
        if metric_name == "ndcg_at_k":
            return evaluation.ndcg_at_k
        if metric_name == "mean_latency_ms":
            return evaluation.mean_latency_ms
        if metric_name == "abstention_accuracy":
            return evaluation.abstention_accuracy

        raise ValueError(f"Unsupported evaluation metric: {metric_name!r}")

    @staticmethod
    def _compare_metric(
        baseline: float,
        candidate: float,
        *,
        higher_is_better: bool,
    ) -> RetrievalEvaluationMetricComparison:
        delta = candidate - baseline

        if delta == 0.0:
            status = RetrievalEvaluationMetricStatus.UNCHANGED
        elif (delta > 0.0) == higher_is_better:
            status = RetrievalEvaluationMetricStatus.IMPROVED
        else:
            status = RetrievalEvaluationMetricStatus.REGRESSED

        return RetrievalEvaluationMetricComparison(
            baseline=baseline,
            candidate=candidate,
            delta=delta,
            status=status,
        )

    @staticmethod
    def _validate_compatibility(
        baseline: RetrievalEvaluationRun,
        candidate: RetrievalEvaluationRun,
    ) -> None:
        baseline_lineage = baseline.lineage
        candidate_lineage = candidate.lineage

        checks = (
            (
                "dataset_name",
                baseline_lineage.dataset_name,
                candidate_lineage.dataset_name,
            ),
            (
                "dataset_version",
                baseline_lineage.dataset_version,
                candidate_lineage.dataset_version,
            ),
            (
                "evaluator_k",
                baseline_lineage.evaluator_k,
                candidate_lineage.evaluator_k,
            ),
            (
                "min_relevance_score",
                baseline_lineage.min_relevance_score,
                candidate_lineage.min_relevance_score,
            ),
            (
                "embedding_identity",
                baseline_lineage.embedding_identity,
                candidate_lineage.embedding_identity,
            ),
        )

        for name, baseline_value, candidate_value in checks:
            if baseline_value != candidate_value:
                raise ValueError(
                    f"Evaluation runs are incompatible for {name}: "
                    f"baseline={baseline_value!r}, "
                    f"candidate={candidate_value!r}."
                )
