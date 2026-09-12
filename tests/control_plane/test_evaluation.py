from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient

from app.control_plane.app import app
from app.control_plane.evaluation_application_service import EvaluationApplicationResult
from app.control_plane.evaluation_service import EvaluationExecutionResult
from app.control_plane.dependencies import (
    get_external_evaluation_release_policy,
    get_retrieval_evaluation_run_store,
    get_retrieval_evaluation_release_decision_store,
)
from rag.evaluation.comparison import (
    RetrievalEvaluationRunComparator,
    RetrievalRegressionPolicy,
)
from rag.evaluation.external import (
    ExternalEvaluationMetricPolicy,
    ExternalEvaluationPolicy,
    ExternalEvaluationReleasePolicy,
    ExternalEvaluationResult,
)
from rag.evaluation.lineage import RetrievalEvaluationLineage
from rag.evaluation.models import RetrievalEvaluationResult
from rag.evaluation.policy import RetrievalEvaluationPolicy
from rag.evaluation.quality_gate import RetrievalQualityGateResult
from rag.evaluation.release import RetrievalEvaluationReleaseGate
from rag.evaluation.composite_release import CompositeEvaluationReleaseGate
from rag.evaluation.external.release import (
    ExternalEvaluationReleaseDecision,
    ExternalEvaluationReleaseGate,
)
from rag.evaluation.run import (
    RetrievalEvaluationRegression,
    RetrievalEvaluationRun,
)
from rag.evaluation.stores.in_memory import (
    InMemoryRetrievalEvaluationReleaseDecisionStore,
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
    external_evaluations: tuple[ExternalEvaluationResult, ...] = (),
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
        external_evaluations=external_evaluations,
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
    *,
    external_release_required: bool = False,
    decision_store: InMemoryRetrievalEvaluationReleaseDecisionStore | None = None,
) -> None:
    if decision_store is None:
        decision_store = InMemoryRetrievalEvaluationReleaseDecisionStore()

    app.dependency_overrides[get_retrieval_evaluation_run_store] = lambda: store
    app.dependency_overrides[get_retrieval_evaluation_release_decision_store] = (
        lambda: decision_store
    )
    app.dependency_overrides[get_external_evaluation_release_policy] = (
        lambda: ExternalEvaluationReleasePolicy(
            name="test-external-release",
            required=external_release_required,
        )
    )


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
    assert payload["external_evaluations"] == []
    assert payload["external_quality_gate"] is None

    assert "query_results" not in payload["evaluation"]
    assert "query" not in payload
    assert "chunks" not in payload
    assert "retrieved_results" not in payload


def test_get_evaluation_run_exposes_external_evaluation_evidence() -> None:
    store = InMemoryRetrievalEvaluationRunStore()
    run = _run(
        "run-external-evaluation",
        created_at=datetime(2026, 9, 11, 12, 0, tzinfo=UTC),
        external_evaluations=(
            ExternalEvaluationResult(
                provider="ragas",
                evaluator="faithfulness",
                metrics={"faithfulness": 0.91},
                evaluated_samples=7,
                metadata={"version": "0.4.3"},
            ),
            ExternalEvaluationResult(
                provider="custom",
                evaluator="answer_relevance",
                metrics={"answer_relevance": 0.88},
                evaluated_samples=7,
            ),
        ),
    )

    external_policy = ExternalEvaluationPolicy(
        name="production-external-rag-v1",
        metrics=(
            ExternalEvaluationMetricPolicy(
                provider="ragas",
                evaluator="faithfulness",
                metric_name="faithfulness",
                minimum_value=0.90,
            ),
            ExternalEvaluationMetricPolicy(
                provider="custom",
                evaluator="answer_relevance",
                metric_name="answer_relevance",
                minimum_value=0.85,
            ),
        ),
    )
    run = run.with_external_quality_gate(external_policy)

    asyncio.run(store.save(run))
    _install_store(store)

    response = client.get(
        "/api/v1/evaluation/runs/run-external-evaluation",
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["external_evaluations"] == [
        {
            "provider": "ragas",
            "evaluator": "faithfulness",
            "metrics": {"faithfulness": 0.91},
            "evaluated_samples": 7,
            "metadata": {"version": "0.4.3"},
        },
        {
            "provider": "custom",
            "evaluator": "answer_relevance",
            "metrics": {"answer_relevance": 0.88},
            "evaluated_samples": 7,
            "metadata": {},
        },
    ]

    assert payload["external_quality_gate"] == {
        "quality_gate_passed": True,
        "quality_gate_policy": "production-external-rag-v1",
        "quality_gate_errors": [],
        "quality_gate_metrics": {
            "ragas/faithfulness/faithfulness": 0.91,
            "custom/answer_relevance/answer_relevance": 0.88,
        },
    }

    # External evidence is informational at this stage and must not
    # alter the native release decision.
    assert payload["passed"] is True
    assert payload["release_passed"] is True


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


def test_composite_release_passes_when_external_evaluation_is_optional() -> None:
    store = InMemoryRetrievalEvaluationRunStore()
    run = _run(
        "composite-optional-pass",
        created_at=datetime.now(UTC),
    )

    asyncio.run(store.save(run))
    _install_store(store, external_release_required=False)

    response = client.get(
        "/api/v1/evaluation/runs/composite-optional-pass/composite-release-decision",
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["run_id"] == "composite-optional-pass"
    assert payload["passed"] is True
    assert payload["errors"] == []
    assert payload["native"]["passed"] is True
    assert payload["external"]["passed"] is True
    assert payload["external"]["errors"] == []
    assert payload["external"]["policy"] == {
        "name": "test-external-release",
        "required": False,
    }


def test_composite_release_fails_when_required_external_evaluation_is_missing() -> None:
    store = InMemoryRetrievalEvaluationRunStore()
    run = _run(
        "composite-required-missing",
        created_at=datetime.now(UTC),
    )

    asyncio.run(store.save(run))
    _install_store(store, external_release_required=True)

    response = client.get(
        "/api/v1/evaluation/runs/composite-required-missing/composite-release-decision",
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["passed"] is False
    assert payload["native"]["passed"] is True
    assert payload["external"]["passed"] is False
    assert payload["external"]["policy"] == {
        "name": "test-external-release",
        "required": True,
    }
    assert "external: required external evaluation evidence is missing" in payload["errors"]


def test_composite_release_passes_when_external_quality_gate_passes() -> None:
    store = InMemoryRetrievalEvaluationRunStore()

    external_result = ExternalEvaluationResult(
        provider="ragas",
        evaluator="faithfulness",
        metrics={"faithfulness": 0.95},
        evaluated_samples=7,
    )

    run = _run(
        "composite-external-pass",
        created_at=datetime.now(UTC),
        external_evaluations=(external_result,),
    )

    external_policy = ExternalEvaluationPolicy(
        name="external-quality-v1",
        metrics=(
            ExternalEvaluationMetricPolicy(
                provider="ragas",
                evaluator="faithfulness",
                metric_name="faithfulness",
                minimum_value=0.90,
            ),
        ),
    )

    run = run.with_external_quality_gate(external_policy)

    asyncio.run(store.save(run))
    _install_store(store, external_release_required=True)

    response = client.get(
        "/api/v1/evaluation/runs/composite-external-pass/composite-release-decision",
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["passed"] is True
    assert payload["errors"] == []
    assert payload["native"]["passed"] is True
    assert payload["external"]["passed"] is True
    assert payload["external"]["errors"] == []


def test_composite_release_fails_when_external_quality_gate_fails() -> None:
    store = InMemoryRetrievalEvaluationRunStore()

    external_result = ExternalEvaluationResult(
        provider="ragas",
        evaluator="faithfulness",
        metrics={"faithfulness": 0.75},
        evaluated_samples=7,
    )

    run = _run(
        "composite-external-fail",
        created_at=datetime.now(UTC),
        external_evaluations=(external_result,),
    )

    external_policy = ExternalEvaluationPolicy(
        name="external-quality-v1",
        metrics=(
            ExternalEvaluationMetricPolicy(
                provider="ragas",
                evaluator="faithfulness",
                metric_name="faithfulness",
                minimum_value=0.90,
            ),
        ),
    )

    run = run.with_external_quality_gate(external_policy)

    asyncio.run(store.save(run))
    _install_store(store, external_release_required=True)

    response = client.get(
        "/api/v1/evaluation/runs/composite-external-fail/composite-release-decision",
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["passed"] is False
    assert payload["native"]["passed"] is True
    assert payload["external"]["passed"] is False
    assert payload["external"]["errors"]
    assert any(error.startswith("external: ") for error in payload["errors"])


def test_composite_release_fails_when_native_release_fails() -> None:
    store = InMemoryRetrievalEvaluationRunStore()
    run = _run(
        "composite-native-fail",
        created_at=datetime.now(UTC),
        passed=False,
        recall=0.5,
    )

    asyncio.run(store.save(run))
    _install_store(store, external_release_required=False)

    response = client.get(
        "/api/v1/evaluation/runs/composite-native-fail/composite-release-decision",
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["passed"] is False
    assert payload["native"]["passed"] is False
    assert "native: retrieval quality gate failed" in payload["errors"]
    assert payload["external"]["passed"] is True


def test_composite_release_accumulates_native_and_external_failures() -> None:
    store = InMemoryRetrievalEvaluationRunStore()

    external_result = ExternalEvaluationResult(
        provider="ragas",
        evaluator="faithfulness",
        metrics={"faithfulness": 0.75},
        evaluated_samples=7,
    )

    run = _run(
        "composite-both-fail",
        created_at=datetime.now(UTC),
        passed=False,
        recall=0.5,
        external_evaluations=(external_result,),
    )

    external_policy = ExternalEvaluationPolicy(
        name="external-quality-v1",
        metrics=(
            ExternalEvaluationMetricPolicy(
                provider="ragas",
                evaluator="faithfulness",
                metric_name="faithfulness",
                minimum_value=0.90,
            ),
        ),
    )

    run = run.with_external_quality_gate(external_policy)

    asyncio.run(store.save(run))
    _install_store(store, external_release_required=True)

    response = client.get(
        "/api/v1/evaluation/runs/composite-both-fail/composite-release-decision",
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["passed"] is False
    assert "native: retrieval quality gate failed" in payload["errors"]
    assert any(error.startswith("external: ") for error in payload["errors"])


def test_composite_release_matches_composite_gate_contract() -> None:
    store = InMemoryRetrievalEvaluationRunStore()

    external_result = ExternalEvaluationResult(
        provider="ragas",
        evaluator="faithfulness",
        metrics={"faithfulness": 0.95},
        evaluated_samples=7,
    )

    run = _run(
        "composite-contract",
        created_at=datetime.now(UTC),
        external_evaluations=(external_result,),
    )

    external_policy = ExternalEvaluationPolicy(
        name="external-quality-v1",
        metrics=(
            ExternalEvaluationMetricPolicy(
                provider="ragas",
                evaluator="faithfulness",
                metric_name="faithfulness",
                minimum_value=0.90,
            ),
        ),
    )

    run = run.with_external_quality_gate(external_policy)

    asyncio.run(store.save(run))
    _install_store(store, external_release_required=True)

    response = client.get(
        "/api/v1/evaluation/runs/composite-contract/composite-release-decision",
        headers=AUTH_HEADERS,
    )

    native = RetrievalEvaluationReleaseGate.evaluate(run)
    external = ExternalEvaluationReleaseGate.evaluate(
        quality_gate=run.external_quality_gate,
        policy=ExternalEvaluationReleasePolicy(
            name="test-external-release",
            required=True,
        ),
    )
    expected = CompositeEvaluationReleaseGate.evaluate(
        run=run,
        native=native,
        external=external,
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["run_id"] == expected.run_id
    assert payload["passed"] == expected.passed
    assert payload["errors"] == list(expected.errors)

    native_payload = expected.native.as_dict()
    assert payload["native"] == {
        **native_payload,
        "errors": list(expected.native.errors),
    }

    external_payload = expected.external.as_dict()
    assert payload["external"] == {
        **external_payload,
        "errors": list(expected.external.errors),
    }


def test_composite_release_returns_persisted_decision_without_recalculation() -> None:
    store = InMemoryRetrievalEvaluationRunStore()
    decision_store = InMemoryRetrievalEvaluationReleaseDecisionStore()

    run = _run(
        "composite-persisted",
        created_at=datetime.now(UTC),
    )

    asyncio.run(store.save(run))

    native = RetrievalEvaluationReleaseGate.evaluate(run)
    persisted_external = ExternalEvaluationReleaseDecision(
        passed=True,
        errors=(),
        policy=ExternalEvaluationReleasePolicy(
            name="historical-external-release",
            required=False,
        ),
    )

    persisted_decision = CompositeEvaluationReleaseGate.evaluate(
        run=run,
        native=native,
        external=persisted_external,
    )

    asyncio.run(decision_store.save(persisted_decision))

    _install_store(
        store,
        external_release_required=True,
        decision_store=decision_store,
    )

    response = client.get(
        "/api/v1/evaluation/runs/composite-persisted/" "composite-release-decision",
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["run_id"] == "composite-persisted"
    assert payload["passed"] is True
    assert payload["errors"] == []
    assert payload["external"]["passed"] is True
    assert payload["external"]["policy"]["required"] is False
    assert payload["external"]["policy"]["name"] == "historical-external-release"


def test_composite_release_missing_run_returns_404() -> None:
    store = InMemoryRetrievalEvaluationRunStore()
    _install_store(store)

    response = client.get(
        "/api/v1/evaluation/runs/missing-composite/composite-release-decision",
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 404
    assert response.json()["detail"] == ("evaluation run not found: missing-composite")


def test_execute_evaluation_run_returns_created_execution_result() -> None:
    from unittest.mock import AsyncMock, Mock

    from app.control_plane.dependencies import get_evaluation_application_service

    execution_service = Mock()
    execution_service.execute = AsyncMock()

    run = _run(
        "api-run-001",
        created_at=datetime.now(UTC),
    )

    decision = CompositeEvaluationReleaseGate.evaluate(
        run=run,
        native=RetrievalEvaluationReleaseGate.evaluate(run),
        external=ExternalEvaluationReleaseGate.evaluate(
            quality_gate=run.external_quality_gate,
            policy=ExternalEvaluationReleasePolicy(
                name="test-external-release",
                required=False,
            ),
        ),
    )

    execution_service.execute.return_value = EvaluationExecutionResult(
        run=run,
        release_decision=decision,
    )

    application_service = Mock()
    application_service.execute = AsyncMock(
        return_value=EvaluationApplicationResult(
            execution=execution_service.execute.return_value,
        )
    )

    app.dependency_overrides[get_evaluation_application_service] = lambda: application_service
    _install_store(InMemoryRetrievalEvaluationRunStore())

    response = client.post(
        "/api/v1/evaluation/runs",
        headers=AUTH_HEADERS,
        json={
            "dataset_name": "vehicle-retrieval",
            "dataset_version": "v2",
            "run_id": "api-run-001",
            "evaluation_policy": {
                "name": "vehicle-quality-v1",
                "min_recall_at_k": 0.9,
                "min_precision_at_k": 0.8,
                "min_mrr": 1.0,
                "min_ndcg_at_k": 0.95,
            },
            "k": 3,
        },
    )

    assert response.status_code == 201

    payload = response.json()

    assert payload["run"]["run_id"] == "api-run-001"
    assert payload["run"]["dataset_name"] == "vehicle-retrieval"
    assert payload["run"]["dataset_version"] == "v2"
    assert payload["run"]["passed"] is True
    assert payload["release_decision"]["run_id"] == "api-run-001"
    assert payload["release_decision"]["passed"] is True

    application_service.execute.assert_awaited_once()

    call = application_service.execute.await_args
    assert call.kwargs["dataset_name"] == "vehicle-retrieval"
    assert call.kwargs["dataset_version"] == "v2"
    assert call.kwargs["run_id"] == "api-run-001"
    assert call.kwargs["k"] == 3
    assert call.kwargs["baseline"] is None
    assert call.kwargs["regression_policy"] is None


def test_execute_evaluation_run_resolves_explicit_baseline() -> None:
    from unittest.mock import AsyncMock, Mock

    from app.control_plane.dependencies import get_evaluation_application_service

    baseline = _run(
        "baseline-api",
        created_at=datetime.now(UTC),
    )

    store = InMemoryRetrievalEvaluationRunStore()

    async def seed() -> None:
        await store.save(baseline)

    asyncio.run(seed())

    application_service = Mock()
    application_service.execute = AsyncMock(
        side_effect=lambda **kwargs: EvaluationApplicationResult(
            execution=EvaluationExecutionResult(
                run=_run(
                    "candidate-api",
                    created_at=datetime.now(UTC),
                ),
                release_decision=CompositeEvaluationReleaseGate.evaluate(
                    run=kwargs["baseline"],
                    native=RetrievalEvaluationReleaseGate.evaluate(kwargs["baseline"]),
                    external=ExternalEvaluationReleaseGate.evaluate(
                        quality_gate=kwargs["baseline"].external_quality_gate,
                        policy=ExternalEvaluationReleasePolicy(
                            name="test-external-release",
                            required=False,
                        ),
                    ),
                ),
            )
        )
    )

    app.dependency_overrides[get_evaluation_application_service] = lambda: application_service
    _install_store(store)

    response = client.post(
        "/api/v1/evaluation/runs",
        headers=AUTH_HEADERS,
        json={
            "dataset_name": "vehicle-retrieval",
            "dataset_version": "v2",
            "run_id": "candidate-api",
            "evaluation_policy": {
                "name": "vehicle-quality-v1",
                "min_recall_at_k": 0.9,
            },
            "k": 3,
            "baseline_run_id": "baseline-api",
            "regression_policy": {
                "name": "vehicle-regression-v1",
                "max_recall_at_k_degradation": 0.05,
            },
        },
    )

    assert response.status_code == 201

    call = application_service.execute.await_args

    assert call.kwargs["baseline"] is baseline
    assert call.kwargs["regression_policy"].name == "vehicle-regression-v1"
    assert call.kwargs["regression_policy"].max_recall_at_k_degradation == 0.05


def test_execute_evaluation_run_missing_baseline_returns_404() -> None:
    from unittest.mock import AsyncMock, Mock

    from app.control_plane.dependencies import get_evaluation_application_service

    application_service = Mock()
    application_service.execute = AsyncMock()

    app.dependency_overrides[get_evaluation_application_service] = lambda: application_service
    _install_store(InMemoryRetrievalEvaluationRunStore())

    response = client.post(
        "/api/v1/evaluation/runs",
        headers=AUTH_HEADERS,
        json={
            "dataset_name": "vehicle-retrieval",
            "dataset_version": "v2",
            "run_id": "candidate-api",
            "evaluation_policy": {
                "name": "vehicle-quality-v1",
                "min_recall_at_k": 0.9,
            },
            "baseline_run_id": "missing-baseline",
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"] == ("baseline evaluation run not found: missing-baseline")
    application_service.execute.assert_not_awaited()


def test_execute_evaluation_run_unknown_dataset_returns_422() -> None:
    from unittest.mock import AsyncMock, Mock

    from app.control_plane.dependencies import get_evaluation_application_service

    application_service = Mock()
    application_service.execute = AsyncMock(
        side_effect=ValueError("evaluation dataset not found: name='does-not-exist', version='v1'")
    )

    app.dependency_overrides[get_evaluation_application_service] = lambda: application_service
    _install_store(InMemoryRetrievalEvaluationRunStore())

    response = client.post(
        "/api/v1/evaluation/runs",
        headers=AUTH_HEADERS,
        json={
            "dataset_name": "does-not-exist",
            "dataset_version": "v1",
            "run_id": "run-invalid-dataset",
            "evaluation_policy": {
                "name": "test-policy",
                "min_recall_at_k": 0.9,
            },
        },
    )

    assert response.status_code == 422
    assert "evaluation dataset not found" in response.json()["detail"]


def test_execute_evaluation_run_requires_authentication() -> None:
    response = client.post(
        "/api/v1/evaluation/runs",
        json={
            "dataset_name": "vehicle-retrieval",
            "dataset_version": "v2",
            "run_id": "unauthorized-run",
            "evaluation_policy": {
                "name": "test-policy",
                "min_recall_at_k": 0.9,
            },
        },
    )

    assert response.status_code == 401
