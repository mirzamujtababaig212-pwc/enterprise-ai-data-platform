from datetime import UTC, datetime, timedelta

import pytest

from rag.evaluation.lineage import RetrievalEvaluationLineage
from rag.evaluation.models import RetrievalEvaluationResult
from rag.evaluation.policy import RetrievalEvaluationPolicy
from rag.evaluation.quality_gate import RetrievalQualityGateResult
from rag.evaluation.run import RetrievalEvaluationRun
from rag.evaluation.stores.in_memory import (
    InMemoryRetrievalEvaluationRunStore,
)


def _run(
    run_id: str,
    created_at: datetime,
) -> RetrievalEvaluationRun:
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
        created_at=created_at,
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
    run = _run("run-001", datetime(2026, 1, 1, tzinfo=UTC))

    await store.save(run)

    assert await store.get("run-001") == run


@pytest.mark.asyncio
async def test_store_replaces_existing_run_with_same_id() -> None:
    store = InMemoryRetrievalEvaluationRunStore()
    first = _run("run-001", datetime(2026, 1, 1, tzinfo=UTC))
    second = _run("run-001", datetime(2026, 1, 2, tzinfo=UTC))

    await store.save(first)
    await store.save(second)

    assert await store.get("run-001") == second


@pytest.mark.asyncio
async def test_list_returns_runs_newest_first() -> None:
    store = InMemoryRetrievalEvaluationRunStore()
    now = datetime.now(UTC)

    older = _run("run-old", now - timedelta(minutes=2))
    newer = _run("run-new", now - timedelta(minutes=1))

    await store.save(older)
    await store.save(newer)

    runs = await store.list()

    assert [run.run_id for run in runs] == ["run-new", "run-old"]


@pytest.mark.asyncio
async def test_list_uses_run_id_as_deterministic_tiebreaker() -> None:
    store = InMemoryRetrievalEvaluationRunStore()
    created_at = datetime.now(UTC)

    await store.save(_run("run-a", created_at))
    await store.save(_run("run-b", created_at))

    runs = await store.list()

    assert [run.run_id for run in runs] == ["run-b", "run-a"]


@pytest.mark.asyncio
async def test_list_supports_pagination() -> None:
    store = InMemoryRetrievalEvaluationRunStore()
    now = datetime.now(UTC)

    for index in range(3):
        await store.save(
            _run(
                f"run-{index}",
                now - timedelta(minutes=index),
            )
        )

    runs = await store.list(limit=1, offset=1)

    assert [run.run_id for run in runs] == ["run-1"]


@pytest.mark.asyncio
async def test_count_returns_total_runs() -> None:
    store = InMemoryRetrievalEvaluationRunStore()
    now = datetime.now(UTC)

    await store.save(_run("run-1", now))
    await store.save(_run("run-2", now))

    assert await store.count() == 2
