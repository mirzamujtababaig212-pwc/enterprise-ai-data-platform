from __future__ import annotations

from datetime import UTC, datetime

import pytest

from rag.evaluation.comparison.baseline_selector import (
    RetrievalEvaluationBaselineSelectionError,
    RetrievalEvaluationBaselineSelector,
)
from rag.evaluation.lineage import RetrievalEvaluationLineage
from rag.evaluation.models import RetrievalEvaluationResult
from rag.evaluation.policy import RetrievalEvaluationPolicy
from rag.evaluation.quality_gate import RetrievalQualityGate
from rag.evaluation.run import RetrievalEvaluationRun


def _run(run_id: str) -> RetrievalEvaluationRun:
    policy = RetrievalEvaluationPolicy(
        min_recall_at_k=0.8,
        name="baseline-policy",
    )

    evaluation = RetrievalEvaluationResult(
        recall_at_k=1.0,
        precision_at_k=1.0,
        mrr=1.0,
        ndcg_at_k=1.0,
        evaluated_queries=1,
        successful_queries=1,
        failed_queries=0,
        mean_latency_ms=10.0,
        query_results=(),
        abstention_accuracy=1.0,
        abstention_evaluated_queries=1,
    )

    return RetrievalEvaluationRun(
        run_id=run_id,
        created_at=datetime(2026, 9, 11, 12, 0, tzinfo=UTC),
        lineage=RetrievalEvaluationLineage(
            dataset_name="vehicle-retrieval",
            dataset_version="v2",
            evaluation_policy_name=policy.name,
            min_recall_at_k=policy.min_recall_at_k,
            min_precision_at_k=policy.min_precision_at_k,
            min_mrr=policy.min_mrr,
            min_ndcg_at_k=policy.min_ndcg_at_k,
            max_mean_latency_ms=policy.max_mean_latency_ms,
            min_abstention_accuracy=policy.min_abstention_accuracy,
            evaluator_k=3,
            min_relevance_score=None,
        ),
        evaluation=evaluation,
        quality_gate=RetrievalQualityGate.evaluate(
            evaluation,
            policy,
        ),
    )


def test_selects_explicit_baseline() -> None:
    candidate = _run("candidate-run")
    baseline = _run("baseline-run")

    selected = RetrievalEvaluationBaselineSelector.select(
        candidate=candidate,
        baselines=(baseline,),
        baseline_run_id="baseline-run",
    )

    assert selected is baseline


def test_rejects_empty_baseline_run_id() -> None:
    candidate = _run("candidate-run")
    baseline = _run("baseline-run")

    with pytest.raises(
        RetrievalEvaluationBaselineSelectionError,
        match="baseline_run_id must not be empty",
    ):
        RetrievalEvaluationBaselineSelector.select(
            candidate=candidate,
            baselines=(baseline,),
            baseline_run_id="   ",
        )


def test_rejects_missing_baseline() -> None:
    candidate = _run("candidate-run")

    with pytest.raises(
        RetrievalEvaluationBaselineSelectionError,
        match="baseline run not found",
    ):
        RetrievalEvaluationBaselineSelector.select(
            candidate=candidate,
            baselines=(),
            baseline_run_id="baseline-run",
        )


def test_rejects_candidate_as_baseline() -> None:
    candidate = _run("candidate-run")

    with pytest.raises(
        RetrievalEvaluationBaselineSelectionError,
        match="candidate run cannot be used as its own baseline",
    ):
        RetrievalEvaluationBaselineSelector.select(
            candidate=candidate,
            baselines=(candidate,),
            baseline_run_id="candidate-run",
        )


def test_rejects_duplicate_baseline_ids() -> None:
    candidate = _run("candidate-run")
    baseline_a = _run("baseline-run")
    baseline_b = _run("baseline-run")

    with pytest.raises(
        RetrievalEvaluationBaselineSelectionError,
        match="multiple baseline runs found",
    ):
        RetrievalEvaluationBaselineSelector.select(
            candidate=candidate,
            baselines=(baseline_a, baseline_b),
            baseline_run_id="baseline-run",
        )


def test_selection_is_independent_of_baseline_order() -> None:
    candidate = _run("candidate-run")
    baseline = _run("baseline-run")
    other = _run("other-run")

    selected = RetrievalEvaluationBaselineSelector.select(
        candidate=candidate,
        baselines=(other, baseline),
        baseline_run_id="baseline-run",
    )

    assert selected is baseline


def test_rejects_incompatible_dataset_version() -> None:
    candidate = _run("candidate-run")
    baseline = _run("baseline-run")

    baseline = RetrievalEvaluationRun(
        run_id=baseline.run_id,
        created_at=baseline.created_at,
        lineage=RetrievalEvaluationLineage(
            dataset_name=baseline.lineage.dataset_name,
            dataset_version="v1",
            evaluation_policy_name=baseline.lineage.evaluation_policy_name,
            min_recall_at_k=baseline.lineage.min_recall_at_k,
            min_precision_at_k=baseline.lineage.min_precision_at_k,
            min_mrr=baseline.lineage.min_mrr,
            min_ndcg_at_k=baseline.lineage.min_ndcg_at_k,
            max_mean_latency_ms=baseline.lineage.max_mean_latency_ms,
            min_abstention_accuracy=baseline.lineage.min_abstention_accuracy,
            evaluator_k=baseline.lineage.evaluator_k,
            min_relevance_score=baseline.lineage.min_relevance_score,
            embedding_identity=baseline.lineage.embedding_identity,
        ),
        evaluation=baseline.evaluation,
        quality_gate=baseline.quality_gate,
    )

    with pytest.raises(
        RetrievalEvaluationBaselineSelectionError,
        match="dataset_version",
    ):
        RetrievalEvaluationBaselineSelector.select(
            candidate=candidate,
            baselines=(baseline,),
            baseline_run_id="baseline-run",
        )


def test_rejects_incompatible_evaluator_k() -> None:
    candidate = _run("candidate-run")
    baseline = _run("baseline-run")

    baseline_lineage = RetrievalEvaluationLineage(
        dataset_name=baseline.lineage.dataset_name,
        dataset_version=baseline.lineage.dataset_version,
        evaluation_policy_name=baseline.lineage.evaluation_policy_name,
        min_recall_at_k=baseline.lineage.min_recall_at_k,
        min_precision_at_k=baseline.lineage.min_precision_at_k,
        min_mrr=baseline.lineage.min_mrr,
        min_ndcg_at_k=baseline.lineage.min_ndcg_at_k,
        max_mean_latency_ms=baseline.lineage.max_mean_latency_ms,
        min_abstention_accuracy=baseline.lineage.min_abstention_accuracy,
        evaluator_k=5,
        min_relevance_score=baseline.lineage.min_relevance_score,
        embedding_identity=baseline.lineage.embedding_identity,
    )

    baseline = RetrievalEvaluationRun(
        run_id=baseline.run_id,
        created_at=baseline.created_at,
        lineage=baseline_lineage,
        evaluation=baseline.evaluation,
        quality_gate=baseline.quality_gate,
    )

    with pytest.raises(
        RetrievalEvaluationBaselineSelectionError,
        match="evaluator_k",
    ):
        RetrievalEvaluationBaselineSelector.select(
            candidate=candidate,
            baselines=(baseline,),
            baseline_run_id="baseline-run",
        )


def test_rejects_incompatible_relevance_score() -> None:
    candidate = _run("candidate-run")
    baseline = _run("baseline-run")

    baseline_lineage = RetrievalEvaluationLineage(
        dataset_name=baseline.lineage.dataset_name,
        dataset_version=baseline.lineage.dataset_version,
        evaluation_policy_name=baseline.lineage.evaluation_policy_name,
        min_recall_at_k=baseline.lineage.min_recall_at_k,
        min_precision_at_k=baseline.lineage.min_precision_at_k,
        min_mrr=baseline.lineage.min_mrr,
        min_ndcg_at_k=baseline.lineage.min_ndcg_at_k,
        max_mean_latency_ms=baseline.lineage.max_mean_latency_ms,
        min_abstention_accuracy=baseline.lineage.min_abstention_accuracy,
        evaluator_k=baseline.lineage.evaluator_k,
        min_relevance_score=0.75,
        embedding_identity=baseline.lineage.embedding_identity,
    )

    baseline = RetrievalEvaluationRun(
        run_id=baseline.run_id,
        created_at=baseline.created_at,
        lineage=baseline_lineage,
        evaluation=baseline.evaluation,
        quality_gate=baseline.quality_gate,
    )

    with pytest.raises(
        RetrievalEvaluationBaselineSelectionError,
        match="min_relevance_score",
    ):
        RetrievalEvaluationBaselineSelector.select(
            candidate=candidate,
            baselines=(baseline,),
            baseline_run_id="baseline-run",
        )


def test_rejects_incompatible_embedding_identity() -> None:
    candidate = _run("candidate-run")
    baseline = _run("baseline-run")

    from rag.models import EmbeddingIdentity

    baseline_lineage = RetrievalEvaluationLineage(
        dataset_name=baseline.lineage.dataset_name,
        dataset_version=baseline.lineage.dataset_version,
        evaluation_policy_name=baseline.lineage.evaluation_policy_name,
        min_recall_at_k=baseline.lineage.min_recall_at_k,
        min_precision_at_k=baseline.lineage.min_precision_at_k,
        min_mrr=baseline.lineage.min_mrr,
        min_ndcg_at_k=baseline.lineage.min_ndcg_at_k,
        max_mean_latency_ms=baseline.lineage.max_mean_latency_ms,
        min_abstention_accuracy=baseline.lineage.min_abstention_accuracy,
        evaluator_k=baseline.lineage.evaluator_k,
        min_relevance_score=baseline.lineage.min_relevance_score,
        embedding_identity=EmbeddingIdentity(
            requested_provider="different-provider",
            requested_model="different-model",
            resolved_provider="different-provider",
            resolved_model="different-model",
            dimension=999,
        ),
    )

    baseline = RetrievalEvaluationRun(
        run_id=baseline.run_id,
        created_at=baseline.created_at,
        lineage=baseline_lineage,
        evaluation=baseline.evaluation,
        quality_gate=baseline.quality_gate,
    )

    with pytest.raises(
        RetrievalEvaluationBaselineSelectionError,
        match="embedding_identity",
    ):
        RetrievalEvaluationBaselineSelector.select(
            candidate=candidate,
            baselines=(baseline,),
            baseline_run_id="baseline-run",
        )
