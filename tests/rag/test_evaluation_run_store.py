from datetime import datetime, timezone

import pytest

from rag.evaluation.lineage import RetrievalEvaluationLineage
from rag.evaluation.models import RetrievalEvaluationResult
from rag.evaluation.policy import RetrievalEvaluationPolicy
from rag.evaluation.quality_gate import RetrievalQualityGateResult
from rag.evaluation.run import RetrievalEvaluationRun
from rag.evaluation.stores import InMemoryRetrievalEvaluationRunStore


def _run(run_id: str) -> RetrievalEvaluationRun:
    lineage = RetrievalEvaluationLineage(
        dataset_name="vehicle-retrieval",
        dataset_version="v2",
        evaluation_policy_name="vehicle-quality-v1",
        min_recall_at_k=0.9,
        min_precision_at_k=None,
        min_mrr=None,
        min_ndcg_at_k=None,
        max_mean_latency_ms=None,
        min_abstention_accuracy=None,
        evaluator_k=3,
        min_relevance_score=None,
    )

    evaluation = RetrievalEvaluationResult(
        recall_at_k=0.0,
        precision_at_k=0.0,
        mrr=0.0,
        ndcg_at_k=0.0,
        evaluated_queries=0,
        successful_queries=0,
        failed_queries=0,
        mean_latency_ms=0.0,
        query_results=(),
    )

    policy = RetrievalEvaluationPolicy(
        name="vehicle-quality-v1",
        min_recall_at_k=0.9,
    )

    quality_gate = RetrievalQualityGateResult(
        passed=True,
        errors=(),
        metrics={},
        policy=policy,
    )

    return RetrievalEvaluationRun(
        run_id=run_id,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        lineage=lineage,
        evaluation=evaluation,
        quality_gate=quality_gate,
    )


@pytest.mark.asyncio
async def test_store_returns_none_for_unknown_run() -> None:
    store = InMemoryRetrievalEvaluationRunStore()

    assert await store.get("missing") is None


@pytest.mark.asyncio
async def test_store_saves_and_retrieves_run() -> None:
    store = InMemoryRetrievalEvaluationRunStore()
    run = _run("run-001")

    await store.save(run)

    assert await store.get("run-001") == run


@pytest.mark.asyncio
async def test_store_replaces_existing_run_with_same_id() -> None:
    store = InMemoryRetrievalEvaluationRunStore()

    first = _run("run-001")
    second = _run("run-001")

    await store.save(first)
    await store.save(second)

    assert await store.get("run-001") == second
