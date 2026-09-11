from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient

from app.control_plane.app import app
from app.control_plane.dependencies import (
    get_retrieval_evaluation_run_store,
)
from rag.evaluation.comparison import (
    RetrievalEvaluationRunComparator,
    RetrievalRegressionPolicy,
)
from rag.evaluation.lineage import RetrievalEvaluationLineage
from rag.evaluation.models import RetrievalEvaluationResult
from rag.evaluation.policy import RetrievalEvaluationPolicy
from rag.evaluation.quality_gate import RetrievalQualityGateResult
from rag.evaluation.release import RetrievalEvaluationReleaseGate
from rag.evaluation.run import (
    RetrievalEvaluationRegression,
    RetrievalEvaluationRun,
)
from rag.evaluation.stores.in_memory import (
    InMemoryRetrievalEvaluationRunStore,
)

client = TestClient(app)

AUTH_HEADERS = {"x-api-key": "super-secret-key"}


def _run(
    run_id: str,
    *,
    created_at: datetime,
    passed: bool = True,
    recall: float = 1.0,
    regression: RetrievalEvaluationRegression | None = None,
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
        recall_at_k=recall,
        precision_at_k=0.8,
        mrr=1.0,
        ndcg_at_k=0.95,
        evaluated_queries=7,
        successful_queries=7,
        failed_queries=0,
        mean_latency_ms=10.0,
        query_results=(),
    )

    policy = RetrievalEvaluationPolicy(
        name="vehicle-quality-v1",
        min_recall_at_k=0.9,
    )

    quality_gate = RetrievalQualityGateResult(
        passed=passed,
        errors=() if passed else ("recall below threshold",),
        metrics={
            "recall_at_k": recall,
        },
        policy=policy,
    )

    return RetrievalEvaluationRun(
        run_id=run_id,
        created_at=created_at,
        lineage=lineage,
        evaluation=evaluation,
        quality_gate=quality_gate,
        regression=regression,
    )


def _regression(
    *,
    baseline: RetrievalEvaluationRun,
    candidate: RetrievalEvaluationRun,
    passed: bool,
) -> RetrievalEvaluationRegression:
    comparison = RetrievalEvaluationRunComparator.compare(
        baseline,
        candidate,
    )

    policy = RetrievalRegressionPolicy(
        name="vehicle-regression-v1",
        max_recall_at_k_degradation=0.0,
    )

    errors = () if passed else ("recall regression exceeds tolerance",)

    from rag.evaluation.comparison.regression_policy import (
        RetrievalRegressionPolicyResult,
    )

    result = RetrievalRegressionPolicyResult(
        passed=passed,
        errors=errors,
        policy_name=policy.name,
    )

    return RetrievalEvaluationRegression(
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        comparison=comparison,
        policy=policy,
        result=result,
    )


def _install_store(
    store: InMemoryRetrievalEvaluationRunStore,
) -> None:
    app.dependency_overrides[get_retrieval_evaluation_run_store] = lambda: store


def teardown_function() -> None:
    app.dependency_overrides.clear()


def test_list_evaluation_runs_returns_paginated_aggregate_evidence() -> None:
    store = InMemoryRetrievalEvaluationRunStore()
    now = datetime.now(UTC)

    older = _run(
        "run-old",
        created_at=now - timedelta(minutes=2),
    )
    newer = _run(
        "run-new",
        created_at=now - timedelta(minutes=1),
    )

    import asyncio

    async def seed() -> None:
        await store.save(older)
        await store.save(newer)

    asyncio.run(seed())
    _install_store(store)

    response = client.get(
        "/api/v1/evaluation/runs",
        params={"limit": 1, "offset": 0},
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["total"] == 2
    assert payload["limit"] == 1
    assert payload["offset"] == 0
    assert len(payload["runs"]) == 1
    assert payload["runs"][0]["run_id"] == "run-new"
    assert payload["runs"][0]["dataset_name"] == "vehicle-retrieval"
    assert payload["runs"][0]["dataset_version"] == "v2"
    assert payload["runs"][0]["release_passed"] is True


def test_get_evaluation_run_returns_aggregate_evidence_only() -> None:
    store = InMemoryRetrievalEvaluationRunStore()
    run = _run(
        "run-001",
        created_at=datetime(2026, 9, 11, 12, 0, tzinfo=UTC),
    )

    import asyncio

    asyncio.run(store.save(run))
    _install_store(store)

    response = client.get(
        "/api/v1/evaluation/runs/run-001",
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["run_id"] == "run-001"
    assert payload["dataset_name"] == "vehicle-retrieval"
    assert payload["dataset_version"] == "v2"
    assert payload["evaluation"]["retrieval_recall_at_k"] == 1.0
    assert payload["quality_gate"]["quality_gate_passed"] is True
    assert payload["passed"] is True
    assert payload["release_passed"] is True

    assert "query_results" not in payload["evaluation"]
    assert "query" not in payload
    assert "chunks" not in payload
    assert "retrieved_results" not in payload


def test_get_missing_evaluation_run_returns_404() -> None:
    store = InMemoryRetrievalEvaluationRunStore()
    _install_store(store)

    response = client.get(
        "/api/v1/evaluation/runs/missing",
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 404
    assert response.json()["detail"] == ("evaluation run not found: missing")


def test_get_evaluation_comparison_returns_persisted_comparison() -> None:
    store = InMemoryRetrievalEvaluationRunStore()
    created_at = datetime(2026, 9, 11, 12, 0, tzinfo=UTC)

    baseline = _run(
        "baseline-001",
        created_at=created_at,
    )
    candidate = _run(
        "candidate-001",
        created_at=created_at + timedelta(minutes=1),
        recall=0.95,
    )

    regression = _regression(
        baseline=baseline,
        candidate=candidate,
        passed=False,
    )

    candidate = _run(
        "candidate-001",
        created_at=created_at + timedelta(minutes=1),
        recall=0.95,
        regression=regression,
    )

    import asyncio

    async def seed() -> None:
        await store.save(candidate)

    asyncio.run(seed())
    _install_store(store)

    response = client.get(
        "/api/v1/evaluation/runs/candidate-001/comparison",
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["baseline_run_id"] == "baseline-001"
    assert payload["candidate_run_id"] == "candidate-001"
    assert "recall_at_k" in payload["metrics"]


def test_get_evaluation_comparison_without_regression_returns_404() -> None:
    store = InMemoryRetrievalEvaluationRunStore()
    run = _run(
        "run-no-regression",
        created_at=datetime.now(UTC),
    )

    import asyncio

    asyncio.run(store.save(run))
    _install_store(store)

    response = client.get(
        "/api/v1/evaluation/runs/run-no-regression/comparison",
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 404
    assert response.json()["detail"] == (
        "evaluation run has no regression comparison: run-no-regression"
    )


def test_release_decision_passes_for_passing_run() -> None:
    store = InMemoryRetrievalEvaluationRunStore()
    run = _run(
        "run-pass",
        created_at=datetime.now(UTC),
    )

    import asyncio

    asyncio.run(store.save(run))
    _install_store(store)

    response = client.get(
        "/api/v1/evaluation/runs/run-pass/release-decision",
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200
    assert response.json() == {
        "run_id": "run-pass",
        "passed": True,
        "errors": [],
    }


def test_release_decision_fails_for_quality_gate_failure() -> None:
    store = InMemoryRetrievalEvaluationRunStore()
    run = _run(
        "run-quality-fail",
        created_at=datetime.now(UTC),
        passed=False,
        recall=0.5,
    )

    import asyncio

    asyncio.run(store.save(run))
    _install_store(store)

    response = client.get(
        "/api/v1/evaluation/runs/run-quality-fail/release-decision",
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["run_id"] == "run-quality-fail"
    assert payload["passed"] is False
    assert "retrieval quality gate failed" in payload["errors"]


def test_release_decision_fails_for_regression_policy_failure() -> None:
    store = InMemoryRetrievalEvaluationRunStore()
    created_at = datetime.now(UTC)

    baseline = _run(
        "baseline-release",
        created_at=created_at,
    )
    candidate = _run(
        "candidate-release",
        created_at=created_at + timedelta(minutes=1),
        recall=0.95,
    )

    regression = _regression(
        baseline=baseline,
        candidate=candidate,
        passed=False,
    )

    candidate = _run(
        "candidate-release",
        created_at=created_at + timedelta(minutes=1),
        recall=0.95,
        regression=regression,
    )

    import asyncio

    asyncio.run(store.save(candidate))
    _install_store(store)

    response = client.get(
        "/api/v1/evaluation/runs/candidate-release/release-decision",
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["passed"] is False
    assert "retrieval regression policy failed" in payload["errors"]


def test_release_gate_matches_existing_release_gate_contract() -> None:
    store = InMemoryRetrievalEvaluationRunStore()
    run = _run(
        "run-contract",
        created_at=datetime.now(UTC),
    )

    import asyncio

    asyncio.run(store.save(run))
    _install_store(store)

    response = client.get(
        "/api/v1/evaluation/runs/run-contract/release-decision",
        headers=AUTH_HEADERS,
    )

    expected = RetrievalEvaluationReleaseGate.evaluate(run)

    assert response.status_code == 200
    assert response.json() == {
        "run_id": expected.run_id,
        "passed": expected.passed,
        "errors": list(expected.errors),
    }
