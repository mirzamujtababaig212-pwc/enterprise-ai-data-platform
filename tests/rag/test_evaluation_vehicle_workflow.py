import pytest

from rag.evaluation.dataset import RetrievalEvaluationDataset
from rag.evaluation.datasets.vehicle import (
    VEHICLE_EMBEDDING_IDENTITY,
    VehicleBenchmarkEmbeddingService,
    vehicle_benchmark_chunks,
    vehicle_evaluation_cases,
)
from rag.evaluation.evaluator import RetrievalEvaluator
from rag.evaluation.policy import RetrievalEvaluationPolicy
from rag.evaluation.workflow import RetrievalEvaluationWorkflow
from rag.retrieval import SemanticRetriever
from rag.stores import InMemoryVectorStore


@pytest.mark.asyncio
async def test_vehicle_benchmark_workflow_preserves_real_retrieval_lineage() -> None:
    vector_store = InMemoryVectorStore()
    await vector_store.upsert(vehicle_benchmark_chunks())

    retriever = SemanticRetriever(
        embedding_service=VehicleBenchmarkEmbeddingService(),
        vector_store=vector_store,
    )

    evaluator = RetrievalEvaluator(
        retriever,
        k=3,
        embedding_identity=VEHICLE_EMBEDDING_IDENTITY,
    )

    policy = RetrievalEvaluationPolicy(
        name="vehicle-retrieval-quality-v2",
        min_recall_at_k=1.0,
        min_precision_at_k=0.8,
        min_mrr=1.0,
        min_ndcg_at_k=0.95,
    )

    dataset = RetrievalEvaluationDataset.from_cases(
        "vehicle-retrieval",
        vehicle_evaluation_cases(),
        version="v2",
    )

    workflow = RetrievalEvaluationWorkflow(
        evaluator=evaluator,
        policy=policy,
    )

    result = await workflow.run(dataset)

    assert result.passed is True
    assert result.evaluation.evaluated_queries == 7
    assert result.evaluation.successful_queries == 7
    assert result.evaluation.failed_queries == 0

    assert result.evaluation.recall_at_k == pytest.approx(1.0)
    assert result.evaluation.precision_at_k == pytest.approx(0.8571428571)
    assert result.evaluation.mrr == pytest.approx(1.0)
    assert result.evaluation.ndcg_at_k == pytest.approx(0.9775, abs=1e-4)

    assert result.lineage.dataset_name == "vehicle-retrieval"
    assert result.lineage.dataset_version == "v2"
    assert result.lineage.evaluation_policy_name == "vehicle-retrieval-quality-v2"
    assert result.lineage.evaluator_k == 3
    assert result.lineage.min_relevance_score is None
    assert result.lineage.embedding_identity == VEHICLE_EMBEDDING_IDENTITY

    lineage = result.as_dict()["lineage"]

    assert lineage["dataset_name"] == "vehicle-retrieval"
    assert lineage["dataset_version"] == "v2"
    assert lineage["evaluation_policy_name"] == "vehicle-retrieval-quality-v2"
    assert lineage["evaluator_k"] == 3
    assert lineage["embedding_requested_provider"] == "test-provider"
    assert lineage["embedding_requested_model"] == "vehicle-benchmark"
    assert lineage["embedding_resolved_provider"] == "test-provider"
    assert lineage["embedding_resolved_model"] == "vehicle-benchmark"
    assert lineage["embedding_dimension"] == 6
