from unittest.mock import AsyncMock, MagicMock

import pytest

from rag.evaluation.external.models import ExternalEvaluationResult
from rag.evaluation.external.workflow import RAGGenerationEvaluationWorkflow
from rag.evaluation.models import RetrievalEvaluationCase
from rag.retrieval.retriever import SemanticRetriever
from rag.stores.in_memory import InMemoryVectorStore


def _semantic_retriever_with_in_memory_store() -> SemanticRetriever:
    retriever = object.__new__(SemanticRetriever)
    retriever.vector_store = InMemoryVectorStore()
    return retriever


@pytest.mark.asyncio
async def test_generation_evaluation_workflow_maps_rag_results_to_external_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retriever = _semantic_retriever_with_in_memory_store()
    chat_service = MagicMock()

    rag_query_service = MagicMock()
    rag_query_service.query = AsyncMock(
        side_effect=[
            MagicMock(
                answer="Electric vehicles use battery-powered electric motors.",
                sources=(
                    MagicMock(
                        content=(
                            "Electric vehicles use battery-powered electric motors "
                            "to provide propulsion."
                        )
                    ),
                ),
            ),
            MagicMock(
                answer="Gasoline vehicles use internal combustion engines.",
                sources=(
                    MagicMock(
                        content=(
                            "Gasoline vehicles use internal combustion engines "
                            "to generate propulsion."
                        )
                    ),
                ),
            ),
        ]
    )

    monkeypatch.setattr(
        "rag.evaluation.external.workflow.RAGQueryService",
        MagicMock(return_value=rag_query_service),
    )

    evaluator = MagicMock()
    evaluator.evaluate = AsyncMock(
        return_value=ExternalEvaluationResult(
            provider="test",
            evaluator="faithfulness",
            metrics={"faithfulness": 1.0},
            evaluated_samples=2,
        )
    )

    workflow = RAGGenerationEvaluationWorkflow(
        chat_service=chat_service,
        evaluator=evaluator,
        top_k=3,
    )

    cases = (
        RetrievalEvaluationCase(
            query="How does an electric vehicle get its power?",
            relevant_chunk_ids=("vehicle-electric-powertrain",),
        ),
        RetrievalEvaluationCase(
            query="How does a gasoline vehicle generate propulsion?",
            relevant_chunk_ids=("vehicle-gasoline-engine",),
        ),
    )

    result = await workflow.evaluate(
        cases,
        retriever=retriever,
    )

    assert result.provider == "test"
    assert result.evaluator == "faithfulness"
    assert result.metrics["faithfulness"] == 1.0
    assert result.evaluated_samples == 2

    assert result.retrieval_artifact is not None
    assert result.retrieval_artifact.as_dict() == {
        "retriever_type": "SemanticRetriever",
        "vector_store_type": "InMemoryVectorStore",
        "hybrid_configuration": None,
        "reranker_configuration": None,
    }

    evaluator.evaluate.assert_awaited_once()

    samples = evaluator.evaluate.await_args.args[0]

    assert len(samples) == 2

    assert samples[0].query == cases[0].query
    assert samples[0].response == ("Electric vehicles use battery-powered electric motors.")
    assert samples[0].retrieved_contexts == (
        "Electric vehicles use battery-powered electric motors " "to provide propulsion.",
    )

    assert samples[1].query == cases[1].query
    assert samples[1].response == ("Gasoline vehicles use internal combustion engines.")


@pytest.mark.asyncio
async def test_generation_evaluation_workflow_reuses_evaluation_retriever(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retriever = _semantic_retriever_with_in_memory_store()
    chat_service = MagicMock()

    rag_query_service = MagicMock()
    rag_query_service.query = AsyncMock(
        return_value=MagicMock(
            answer="The answer.",
            sources=(MagicMock(content="Retrieved context."),),
        )
    )

    query_service_constructor = MagicMock(return_value=rag_query_service)

    monkeypatch.setattr(
        "rag.evaluation.external.workflow.RAGQueryService",
        query_service_constructor,
    )

    evaluator = MagicMock()
    evaluator.evaluate = AsyncMock(
        return_value=ExternalEvaluationResult(
            provider="test",
            evaluator="faithfulness",
            metrics={"faithfulness": 1.0},
            evaluated_samples=1,
        )
    )

    workflow = RAGGenerationEvaluationWorkflow(
        chat_service=chat_service,
        evaluator=evaluator,
        top_k=7,
        min_score=0.75,
    )

    case = RetrievalEvaluationCase(
        query="What is this?",
        relevant_chunk_ids=("chunk-1",),
    )

    await workflow.evaluate(
        (case,),
        retriever=retriever,
    )

    query_service_constructor.assert_called_once_with(
        retriever=retriever,
        chat_service=chat_service,
    )

    rag_query_service.query.assert_awaited_once_with(
        "What is this?",
        top_k=7,
        min_score=0.75,
    )


@pytest.mark.asyncio
async def test_generation_evaluation_workflow_rejects_empty_cases() -> None:
    workflow = RAGGenerationEvaluationWorkflow(
        chat_service=MagicMock(),
        evaluator=MagicMock(),
    )

    with pytest.raises(ValueError, match="cases must not be empty"):
        await workflow.evaluate(
            (),
            retriever=MagicMock(),
        )


def test_generation_evaluation_workflow_rejects_invalid_top_k() -> None:
    with pytest.raises(ValueError, match="top_k must be greater than zero"):
        RAGGenerationEvaluationWorkflow(
            chat_service=MagicMock(),
            evaluator=MagicMock(),
            top_k=0,
        )
