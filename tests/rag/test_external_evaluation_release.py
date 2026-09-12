from __future__ import annotations

from rag.evaluation.external import (
    ExternalEvaluationMetricPolicy,
    ExternalEvaluationPolicy,
    ExternalEvaluationQualityGate,
    ExternalEvaluationReleaseGate,
    ExternalEvaluationReleasePolicy,
    ExternalEvaluationResult,
)


def _quality_gate(*, passed: bool = True):
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


def test_optional_external_evaluation_passes_without_evidence() -> None:
    decision = ExternalEvaluationReleaseGate.evaluate(
        quality_gate=None,
        policy=ExternalEvaluationReleasePolicy(
            name="optional-external-v1",
            required=False,
        ),
    )

    assert decision.passed is True
    assert decision.errors == ()


def test_required_external_evaluation_fails_without_evidence() -> None:
    decision = ExternalEvaluationReleaseGate.evaluate(
        quality_gate=None,
        policy=ExternalEvaluationReleasePolicy(
            name="required-external-v1",
            required=True,
        ),
    )

    assert decision.passed is False
    assert decision.errors == ("required external evaluation evidence is missing",)


def test_external_release_passes_when_quality_gate_passes() -> None:
    quality_gate = _quality_gate(passed=True)

    decision = ExternalEvaluationReleaseGate.evaluate(
        quality_gate=quality_gate,
        policy=ExternalEvaluationReleasePolicy(
            name="required-external-v1",
            required=True,
        ),
    )

    assert decision.passed is True
    assert decision.errors == ()


def test_external_release_fails_when_quality_gate_fails() -> None:
    quality_gate = _quality_gate(passed=False)

    assert quality_gate.passed is False

    decision = ExternalEvaluationReleaseGate.evaluate(
        quality_gate=quality_gate,
        policy=ExternalEvaluationReleasePolicy(
            name="required-external-v1",
            required=True,
        ),
    )

    assert decision.passed is False
    assert decision.errors == ("external_faithfulness=0.7000 is below required minimum 0.9000",)


def test_external_release_decision_serializes() -> None:
    decision = ExternalEvaluationReleaseGate.evaluate(
        quality_gate=None,
        policy=ExternalEvaluationReleasePolicy(
            name="optional-external-v1",
            required=False,
        ),
    )

    assert decision.as_dict() == {
        "passed": True,
        "errors": (),
        "policy": {
            "name": "optional-external-v1",
            "required": False,
        },
    }
