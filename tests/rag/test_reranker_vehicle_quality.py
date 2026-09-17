import pytest

from rag.evaluation.dataset_registry import (
    VehicleRetrievalQualityEvaluationDatasetDefinition,
)
from rag.evaluation.evaluator import RetrievalEvaluator
from rag.retrieval import RerankingRetriever, TokenOverlapReranker


@pytest.mark.asyncio
async def test_token_overlap_reranker_vehicle_quality_baseline() -> None:
    definition = VehicleRetrievalQualityEvaluationDatasetDefinition()

    dataset = definition.build_dataset()
    hybrid = await definition.build_retriever()

    reranked = RerankingRetriever(
        retriever=hybrid,
        reranker=TokenOverlapReranker(),
        candidate_k=5,
    )

    evaluator = RetrievalEvaluator(
        reranked,
        k=5,
        embedding_identity=definition.build_embedding_identity(),
    )

    result = await evaluator.evaluate(dataset.cases)

    assert result.evaluated_queries == 16
    assert result.successful_queries == 16
    assert result.failed_queries == 0

    assert result.recall_at_k == pytest.approx(1.0)
    assert result.precision_at_k == pytest.approx(0.2875)
    assert result.mrr == pytest.approx(0.921875)
    assert result.ndcg_at_k == pytest.approx(0.920820, abs=1e-6)


@pytest.mark.asyncio
async def test_token_overlap_reranker_can_degrade_semantic_ranking() -> None:
    definition = VehicleRetrievalQualityEvaluationDatasetDefinition()

    dataset = definition.build_dataset()
    hybrid = await definition.build_retriever()

    reranked = RerankingRetriever(
        retriever=hybrid,
        reranker=TokenOverlapReranker(),
        candidate_k=5,
    )

    evaluator = RetrievalEvaluator(
        reranked,
        k=5,
        embedding_identity=definition.build_embedding_identity(),
    )

    result = await evaluator.evaluate(dataset.cases)

    query_result = next(
        item
        for item in result.query_results
        if item.query == "What equipment supplies electricity to an EV battery?"
    )

    assert query_result.retrieved_chunk_ids == (
        "quality-electric-powertrain",
        "quality-electric-inverter",
        "quality-hybrid-battery",
        "quality-battery-charging",
        "quality-battery-management",
    )

    assert query_result.reciprocal_rank == pytest.approx(0.25)
    assert query_result.ndcg_at_k == pytest.approx(0.430677, abs=1e-6)
