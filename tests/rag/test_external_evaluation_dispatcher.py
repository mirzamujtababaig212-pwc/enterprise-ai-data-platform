from unittest.mock import AsyncMock, MagicMock

import pytest

from rag.evaluation.external.models import (
    ExternalEvaluationRequest,
    ExternalEvaluationResult,
)


@pytest.mark.asyncio
async def test_dispatcher_contract_can_execute_external_evaluation():
    dispatcher = MagicMock()
    dispatcher.evaluate = AsyncMock(
        return_value=ExternalEvaluationResult(
            provider="ragas",
            evaluator="faithfulness",
            metrics={"faithfulness": 0.9},
            evaluated_samples=2,
        )
    )

    request = ExternalEvaluationRequest(
        provider="ragas",
        evaluator="faithfulness",
    )

    cases = [MagicMock(), MagicMock()]
    retriever = MagicMock()

    result = await dispatcher.evaluate(
        request,
        cases=cases,
        retriever=retriever,
    )

    assert result.provider == "ragas"
    assert result.evaluator == "faithfulness"
    assert result.metrics["faithfulness"] == 0.9
    assert result.evaluated_samples == 2

    dispatcher.evaluate.assert_awaited_once_with(
        request,
        cases=cases,
        retriever=retriever,
    )
