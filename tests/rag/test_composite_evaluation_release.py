from __future__ import annotations

from datetime import UTC, datetime

from rag.evaluation.composite_release import (
    CompositeEvaluationReleaseGate,
)
from rag.evaluation.external import (
    ExternalEvaluationMetricPolicy,
    ExternalEvaluationPolicy,
    ExternalEvaluationQualityGate,
    ExternalEvaluationReleaseGate,
    ExternalEvaluationReleasePolicy,
    ExternalEvaluationResult,
)
from rag.evaluation.release import RetrievalEvaluationReleaseGate
from rag.evaluation.run import RetrievalEvaluationRun


def _run() -> RetrievalEvaluationRun:
    return RetrievalEvaluationRun(
        run_id="run-001",
        created_at=datetime.now(UTC),
        lineage=_lineage(),
        evaluation=_evaluation(),
        quality_gate=_native_quality_gate(),
    )


def _lineage():
    from rag.evaluation.lineage import RetrievalEvaluationLineage

    return RetrievalEvaluationLineage(
        dataset_name="vehicle-retrieval",
        dataset_version="v1",
        evaluator_k=3,
        min_relevance_score=0.5,
        evaluation_policy_name="vehicle-retrieval-policy-v1",
        min_recall_at_k=0.9,
        min_precision_at_k=0.9,
        min_mrr=0.9,
        min_ndcg_at_k=0.9,
        max_mean_latency_ms=100.0,
        min_abstention_accuracy=0.9,
        embedding_identity=None,
    )


def _evaluation():
    from rag.evaluation.models import RetrievalEvaluationResult

    return RetrievalEvaluationResult(
        evaluated_queries=1,
        successful_queries=1,
        failed_queries=0,
        query_results=(),
        recall_at_k=1.0,
        precision_at_k=1.0,
        mrr=1.0,
        ndcg_at_k=1.0,
        mean_latency_ms=10.0,
        abstention_accuracy=1.0,
    )


def _native_quality_gate():
    from rag.evaluation.quality_gate import RetrievalQualityGateResult

    return RetrievalQualityGateResult(
        passed=True,
        errors=(),
        policy=_native_policy(),
        metrics={
            "recall_at_k": 1.0,
            "precision_at_k": 1.0,
            "mrr": 1.0,
            "ndcg_at_k": 1.0,
            "mean_latency_ms": 10.0,
            "abstention_accuracy": 1.0,
        },
    )


def _native_policy():
    from rag.evaluation.policy import RetrievalEvaluationPolicy

    return RetrievalEvaluationPolicy(
        min_recall_at_k=0.9,
        min_precision_at_k=0.9,
        min_mrr=0.9,
        min_ndcg_at_k=0.9,
        max_mean_latency_ms=100.0,
        min_abstention_accuracy=0.9,
    )


def _external_quality_gate(*, passed: bool):
    result = ExternalEvaluationResult(
        provider="ragas",
        evaluator="faithfulness",
        metrics={"faithfulness": 0.95 if passed else 0.70},
        evaluated_samples=5,
    )

    policy = ExternalEvaluationPolicy(
        name="external-rag-v1",
        metrics=(
            ExternalEvaluationMetricPolicy(
                provider="ragas",
                evaluator="faithfulness",
                metric_name="faithfulness",
                minimum_value=0.90,
            ),
        ),
    )

    return ExternalEvaluationQualityGate.evaluate(
        (result,),
        policy,
    )


def test_composite_release_passes_when_native_and_external_pass() -> None:
    run = _run()

    native = RetrievalEvaluationReleaseGate.evaluate(run)

    external = ExternalEvaluationReleaseGate.evaluate(
        quality_gate=_external_quality_gate(passed=True),
        policy=ExternalEvaluationReleasePolicy(
            name="required-external-v1",
            required=True,
        ),
    )

    decision = CompositeEvaluationReleaseGate.evaluate(
        run=run,
        native=native,
        external=external,
    )

    assert decision.passed is True
    assert decision.errors == ()
    assert decision.run_id == "run-001"


def test_composite_release_fails_when_native_release_fails() -> None:
    run = _run()

    native = RetrievalEvaluationReleaseGate.evaluate(run)

    # Construct an explicit failed native decision so this test isolates
    # the composite gate rather than the native evaluation implementation.
    from rag.evaluation.release import RetrievalEvaluationReleaseDecision

    native = RetrievalEvaluationReleaseDecision(
        run_id=run.run_id,
        passed=False,
        errors=("retrieval quality gate failed",),
    )

    external = ExternalEvaluationReleaseGate.evaluate(
        quality_gate=_external_quality_gate(passed=True),
        policy=ExternalEvaluationReleasePolicy(
            name="required-external-v1",
            required=True,
        ),
    )

    decision = CompositeEvaluationReleaseGate.evaluate(
        run=run,
        native=native,
        external=external,
    )

    assert decision.passed is False
    assert decision.errors == ("native: retrieval quality gate failed",)


def test_composite_release_fails_when_external_release_fails() -> None:
    run = _run()

    native = RetrievalEvaluationReleaseGate.evaluate(run)

    external = ExternalEvaluationReleaseGate.evaluate(
        quality_gate=_external_quality_gate(passed=False),
        policy=ExternalEvaluationReleasePolicy(
            name="required-external-v1",
            required=True,
        ),
    )

    decision = CompositeEvaluationReleaseGate.evaluate(
        run=run,
        native=native,
        external=external,
    )

    assert decision.passed is False
    assert decision.errors == (
        "external: external_faithfulness=0.7000 is below required minimum 0.9000",
    )


def test_composite_release_accumulates_native_and_external_failures() -> None:
    run = _run()

    from rag.evaluation.release import RetrievalEvaluationReleaseDecision

    native = RetrievalEvaluationReleaseDecision(
        run_id=run.run_id,
        passed=False,
        errors=("retrieval regression policy failed",),
    )

    external = ExternalEvaluationReleaseGate.evaluate(
        quality_gate=None,
        policy=ExternalEvaluationReleasePolicy(
            name="required-external-v1",
            required=True,
        ),
    )

    decision = CompositeEvaluationReleaseGate.evaluate(
        run=run,
        native=native,
        external=external,
    )

    assert decision.passed is False
    assert decision.errors == (
        "native: retrieval regression policy failed",
        "external: required external evaluation evidence is missing",
    )


def test_composite_release_decision_serializes() -> None:
    run = _run()
    native = RetrievalEvaluationReleaseGate.evaluate(run)

    external = ExternalEvaluationReleaseGate.evaluate(
        quality_gate=None,
        policy=ExternalEvaluationReleasePolicy(
            name="optional-external-v1",
            required=False,
        ),
    )

    decision = CompositeEvaluationReleaseGate.evaluate(
        run=run,
        native=native,
        external=external,
    )

    payload = decision.as_dict()

    assert payload["run_id"] == "run-001"
    assert payload["passed"] is True
    assert payload["errors"] == ()
    assert payload["native"]["run_id"] == "run-001"
    assert payload["external"]["passed"] is True


def test_composite_release_rejects_mismatched_native_run() -> None:
    run = _run()

    from rag.evaluation.release import RetrievalEvaluationReleaseDecision

    native = RetrievalEvaluationReleaseDecision(
        run_id="different-run",
        passed=True,
        errors=(),
    )

    external = ExternalEvaluationReleaseGate.evaluate(
        quality_gate=None,
        policy=ExternalEvaluationReleasePolicy(
            name="optional-external-v1",
            required=False,
        ),
    )

    try:
        CompositeEvaluationReleaseGate.evaluate(
            run=run,
            native=native,
            external=external,
        )
    except ValueError as exc:
        assert str(exc) == ("native release decision run_id must match evaluation run")
    else:
        raise AssertionError("expected ValueError")
