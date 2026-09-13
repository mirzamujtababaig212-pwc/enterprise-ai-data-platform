from unittest.mock import AsyncMock, MagicMock

import pytest

from rag.evaluation.external.dispatcher import RagasExternalEvaluationDispatcher
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


@pytest.mark.asyncio
async def test_dispatcher_routes_ragas_faithfulness_to_workflow():
    expected_result = ExternalEvaluationResult(
        provider="ragas",
        evaluator="faithfulness",
        metrics={"faithfulness": 0.9},
        evaluated_samples=2,
    )

    workflow = MagicMock()
    workflow.evaluate = AsyncMock(return_value=expected_result)

    dispatcher = RagasExternalEvaluationDispatcher(
        faithfulness_workflow=workflow,
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

    assert result == expected_result

    workflow.evaluate.assert_awaited_once_with(
        cases,
        retriever=retriever,
    )


@pytest.mark.asyncio
async def test_dispatcher_rejects_unsupported_external_evaluation():
    workflow = MagicMock()
    workflow.evaluate = AsyncMock()

    dispatcher = RagasExternalEvaluationDispatcher(
        faithfulness_workflow=workflow,
    )

    request = ExternalEvaluationRequest(
        provider="ragas",
        evaluator="answer_relevance",
    )

    with pytest.raises(
        ValueError,
        match="Unsupported external evaluation: ragas/answer_relevance",
    ):
        await dispatcher.evaluate(
            request,
            cases=[MagicMock()],
            retriever=MagicMock(),
        )

    workflow.evaluate.assert_not_awaited()


@pytest.mark.asyncio
async def test_dispatcher_does_not_create_workflow_for_unsupported_evaluation():
    factory = MagicMock()

    dispatcher = RagasExternalEvaluationDispatcher(
        faithfulness_workflow_factory=factory,
    )

    request = ExternalEvaluationRequest(
        provider="ragas",
        evaluator="answer_relevance",
    )

    with pytest.raises(
        ValueError,
        match="Unsupported external evaluation: ragas/answer_relevance",
    ):
        await dispatcher.evaluate(
            request,
            cases=[MagicMock()],
            retriever=MagicMock(),
        )

    factory.assert_not_called()


@pytest.mark.asyncio
async def test_dispatcher_lazily_creates_workflow_for_supported_evaluation():
    expected_result = ExternalEvaluationResult(
        provider="ragas",
        evaluator="faithfulness",
        metrics={"faithfulness": 0.9},
        evaluated_samples=1,
    )

    workflow = MagicMock()
    workflow.evaluate = AsyncMock(return_value=expected_result)

    factory = MagicMock(return_value=workflow)

    dispatcher = RagasExternalEvaluationDispatcher(
        faithfulness_workflow_factory=factory,
    )

    request = ExternalEvaluationRequest(
        provider="ragas",
        evaluator="faithfulness",
    )

    cases = [MagicMock()]
    retriever = MagicMock()

    result = await dispatcher.evaluate(
        request,
        cases=cases,
        retriever=retriever,
    )

    assert result == expected_result
    factory.assert_called_once()
    workflow.evaluate.assert_awaited_once_with(
        cases,
        retriever=retriever,
    )


@pytest.mark.asyncio
async def test_dispatcher_reuses_lazily_created_workflow():
    workflow = MagicMock()
    workflow.evaluate = AsyncMock(
        side_effect=[
            ExternalEvaluationResult(
                provider="ragas",
                evaluator="faithfulness",
                metrics={"faithfulness": 0.9},
                evaluated_samples=1,
            ),
            ExternalEvaluationResult(
                provider="ragas",
                evaluator="faithfulness",
                metrics={"faithfulness": 0.8},
                evaluated_samples=1,
            ),
        ]
    )

    factory = MagicMock(return_value=workflow)

    dispatcher = RagasExternalEvaluationDispatcher(
        faithfulness_workflow_factory=factory,
    )

    request = ExternalEvaluationRequest(
        provider="ragas",
        evaluator="faithfulness",
    )

    await dispatcher.evaluate(
        request,
        cases=[MagicMock()],
        retriever=MagicMock(),
    )
    await dispatcher.evaluate(
        request,
        cases=[MagicMock()],
        retriever=MagicMock(),
    )

    factory.assert_called_once()
    assert workflow.evaluate.await_count == 2


@pytest.mark.asyncio
async def test_dispatcher_routes_ragas_answer_relevancy_to_workflow():
    expected_result = ExternalEvaluationResult(
        provider="ragas",
        evaluator="answer_relevancy",
        metrics={"answer_relevancy": 0.85},
        evaluated_samples=2,
    )

    workflow = MagicMock()
    workflow.evaluate = AsyncMock(return_value=expected_result)

    dispatcher = RagasExternalEvaluationDispatcher(
        answer_relevancy_workflow=workflow,
    )

    request = ExternalEvaluationRequest(
        provider="ragas",
        evaluator="answer_relevancy",
    )
    cases = [MagicMock(), MagicMock()]
    retriever = MagicMock()

    result = await dispatcher.evaluate(
        request,
        cases=cases,
        retriever=retriever,
    )

    assert result == expected_result

    workflow.evaluate.assert_awaited_once_with(
        cases,
        retriever=retriever,
    )


@pytest.mark.asyncio
async def test_dispatcher_lazily_creates_answer_relevancy_workflow():
    expected_result = ExternalEvaluationResult(
        provider="ragas",
        evaluator="answer_relevancy",
        metrics={"answer_relevancy": 0.85},
        evaluated_samples=1,
    )

    workflow = MagicMock()
    workflow.evaluate = AsyncMock(return_value=expected_result)

    factory = MagicMock(return_value=workflow)

    dispatcher = RagasExternalEvaluationDispatcher(
        answer_relevancy_workflow_factory=factory,
    )

    request = ExternalEvaluationRequest(
        provider="ragas",
        evaluator="answer_relevancy",
    )

    cases = [MagicMock()]
    retriever = MagicMock()

    result = await dispatcher.evaluate(
        request,
        cases=cases,
        retriever=retriever,
    )

    assert result == expected_result
    factory.assert_called_once()

    workflow.evaluate.assert_awaited_once_with(
        cases,
        retriever=retriever,
    )


@pytest.mark.asyncio
async def test_dispatcher_reuses_lazily_created_answer_relevancy_workflow():
    workflow = MagicMock()
    workflow.evaluate = AsyncMock(
        side_effect=[
            ExternalEvaluationResult(
                provider="ragas",
                evaluator="answer_relevancy",
                metrics={"answer_relevancy": 0.85},
                evaluated_samples=1,
            ),
            ExternalEvaluationResult(
                provider="ragas",
                evaluator="answer_relevancy",
                metrics={"answer_relevancy": 0.75},
                evaluated_samples=1,
            ),
        ]
    )

    factory = MagicMock(return_value=workflow)

    dispatcher = RagasExternalEvaluationDispatcher(
        answer_relevancy_workflow_factory=factory,
    )

    request = ExternalEvaluationRequest(
        provider="ragas",
        evaluator="answer_relevancy",
    )

    await dispatcher.evaluate(
        request,
        cases=[MagicMock()],
        retriever=MagicMock(),
    )

    await dispatcher.evaluate(
        request,
        cases=[MagicMock()],
        retriever=MagicMock(),
    )

    factory.assert_called_once()
    assert workflow.evaluate.await_count == 2


def test_dispatcher_rejects_duplicate_faithfulness_configuration():
    workflow = MagicMock()
    factory = MagicMock()

    with pytest.raises(
        ValueError,
        match="faithfulness_workflow and faithfulness_workflow_factory",
    ):
        RagasExternalEvaluationDispatcher(
            faithfulness_workflow=workflow,
            faithfulness_workflow_factory=factory,
        )


def test_dispatcher_rejects_duplicate_answer_relevancy_configuration():
    workflow = MagicMock()
    factory = MagicMock()

    with pytest.raises(
        ValueError,
        match="answer_relevancy_workflow and " "answer_relevancy_workflow_factory",
    ):
        RagasExternalEvaluationDispatcher(
            answer_relevancy_workflow=workflow,
            answer_relevancy_workflow_factory=factory,
        )


def test_dispatcher_requires_at_least_one_workflow_configuration():
    with pytest.raises(
        ValueError,
        match="At least one RAGAS evaluation workflow",
    ):
        RagasExternalEvaluationDispatcher()
