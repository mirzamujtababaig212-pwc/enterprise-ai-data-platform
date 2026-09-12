import pytest

from rag.evaluation.external.models import ExternalEvaluationRequest


def test_external_evaluation_request():
    request = ExternalEvaluationRequest(
        provider="ragas",
        evaluator="faithfulness",
    )

    assert request.provider == "ragas"
    assert request.evaluator == "faithfulness"


@pytest.mark.parametrize(
    ("provider", "evaluator"),
    [
        ("", "faithfulness"),
        ("ragas", ""),
        ("   ", "faithfulness"),
        ("ragas", "   "),
    ],
)
def test_external_evaluation_request_rejects_empty_values(
    provider,
    evaluator,
):
    with pytest.raises(ValueError):
        ExternalEvaluationRequest(
            provider=provider,
            evaluator=evaluator,
        )
