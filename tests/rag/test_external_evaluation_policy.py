import pytest

from rag.evaluation.external import (
    ExternalEvaluationMetricPolicy,
    ExternalEvaluationPolicy,
    ExternalEvaluationQualityGate,
    ExternalEvaluationResult,
)


def _faithfulness_policy(
    *,
    minimum_value: float | None = 0.90,
    maximum_value: float | None = None,
    required: bool = True,
) -> ExternalEvaluationMetricPolicy:
    return ExternalEvaluationMetricPolicy(
        provider="ragas",
        evaluator="faithfulness",
        metric_name="faithfulness",
        minimum_value=minimum_value,
        maximum_value=maximum_value,
        required=required,
    )


def _faithfulness_result(value: float) -> ExternalEvaluationResult:
    return ExternalEvaluationResult(
        provider="ragas",
        evaluator="faithfulness",
        metrics={"faithfulness": value},
        evaluated_samples=10,
        metadata={"version": "0.4.3"},
    )


def test_metric_policy_requires_threshold() -> None:
    with pytest.raises(
        ValueError,
        match="at least one of minimum_value or maximum_value",
    ):
        _faithfulness_policy(minimum_value=None)


def test_metric_policy_rejects_inverted_range() -> None:
    with pytest.raises(
        ValueError,
        match="minimum_value must not exceed maximum_value",
    ):
        _faithfulness_policy(
            minimum_value=0.95,
            maximum_value=0.90,
        )


def test_external_policy_rejects_empty_metrics() -> None:
    with pytest.raises(
        ValueError,
        match="at least one external evaluation metric policy",
    ):
        ExternalEvaluationPolicy(metrics=())


def test_external_policy_rejects_duplicate_metric_policies() -> None:
    metric = _faithfulness_policy()

    with pytest.raises(
        ValueError,
        match="duplicate external evaluation metric policies",
    ):
        ExternalEvaluationPolicy(metrics=(metric, metric))


def test_quality_gate_passes_minimum_threshold() -> None:
    policy = ExternalEvaluationPolicy(
        name="ragas-faithfulness-v1",
        metrics=(_faithfulness_policy(),),
    )

    result = ExternalEvaluationQualityGate.evaluate(
        (_faithfulness_result(0.91),),
        policy,
    )

    assert result.passed is True
    assert result.errors == ()
    assert result.metrics == {
        "ragas/faithfulness/faithfulness": 0.91,
    }


def test_quality_gate_fails_below_minimum_threshold() -> None:
    policy = ExternalEvaluationPolicy(
        metrics=(_faithfulness_policy(),),
    )

    result = ExternalEvaluationQualityGate.evaluate(
        (_faithfulness_result(0.89),),
        policy,
    )

    assert result.passed is False
    assert result.errors == ("external_faithfulness=0.8900 is below required minimum 0.9000",)


def test_quality_gate_supports_maximum_threshold() -> None:
    policy = ExternalEvaluationPolicy(
        metrics=(
            _faithfulness_policy(
                minimum_value=None,
                maximum_value=0.95,
            ),
        ),
    )

    result = ExternalEvaluationQualityGate.evaluate(
        (_faithfulness_result(0.96),),
        policy,
    )

    assert result.passed is False
    assert result.errors == ("external_faithfulness=0.9600 exceeds maximum 0.9500",)


def test_quality_gate_fails_required_missing_evidence() -> None:
    policy = ExternalEvaluationPolicy(
        metrics=(_faithfulness_policy(),),
    )

    result = ExternalEvaluationQualityGate.evaluate(
        (),
        policy,
    )

    assert result.passed is False
    assert result.errors == (
        "required external evaluation evidence is missing: "
        "provider='ragas', evaluator='faithfulness', metric='faithfulness'",
    )


def test_quality_gate_allows_optional_missing_evidence() -> None:
    policy = ExternalEvaluationPolicy(
        metrics=(
            _faithfulness_policy(
                required=False,
            ),
        ),
    )

    result = ExternalEvaluationQualityGate.evaluate(
        (),
        policy,
    )

    assert result.passed is True
    assert result.errors == ()


def test_quality_gate_fails_required_missing_metric() -> None:
    policy = ExternalEvaluationPolicy(
        metrics=(_faithfulness_policy(),),
    )

    result = ExternalEvaluationQualityGate.evaluate(
        (
            ExternalEvaluationResult(
                provider="ragas",
                evaluator="faithfulness",
                metrics={"answer_relevance": 0.91},
                evaluated_samples=10,
            ),
        ),
        policy,
    )

    assert result.passed is False
    assert result.errors == (
        "required external evaluation metric is missing: "
        "provider='ragas', evaluator='faithfulness', metric='faithfulness'",
    )


def test_quality_gate_as_dict_is_serializable() -> None:
    policy = ExternalEvaluationPolicy(
        name="ragas-faithfulness-v1",
        metrics=(_faithfulness_policy(),),
    )

    result = ExternalEvaluationQualityGate.evaluate(
        (_faithfulness_result(0.91),),
        policy,
    )

    assert result.as_dict() == {
        "quality_gate_passed": True,
        "quality_gate_policy": "ragas-faithfulness-v1",
        "quality_gate_errors": (),
        "quality_gate_metrics": {
            "ragas/faithfulness/faithfulness": 0.91,
        },
    }
