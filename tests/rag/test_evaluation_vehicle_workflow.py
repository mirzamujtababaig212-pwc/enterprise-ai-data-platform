import pytest

from rag.evaluation.dataset_registry import VehicleRetrievalEvaluationDatasetDefinition
from rag.evaluation.evaluator import RetrievalEvaluator
from rag.evaluation.policy import RetrievalEvaluationPolicy
from rag.evaluation.workflow import RetrievalEvaluationWorkflow


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("backend", "expected_vector_store_type"),
    [
        ("in_memory", "InMemoryVectorStore"),
        ("faiss", "FAISSVectorStore"),
    ],
)
async def test_vehicle_benchmark_workflow_preserves_real_retrieval_lineage(
    backend: str,
    expected_vector_store_type: str,
) -> None:
    definition = VehicleRetrievalEvaluationDatasetDefinition(
        vector_store_backend=backend,
    )

    retriever = await definition.build_retriever()

    evaluator = RetrievalEvaluator(
        retriever,
        k=3,
        embedding_identity=definition.build_embedding_identity(),
    )

    policy = RetrievalEvaluationPolicy(
        name="vehicle-retrieval-quality-v2",
        min_recall_at_k=1.0,
        min_precision_at_k=0.8,
        min_mrr=1.0,
        min_ndcg_at_k=0.95,
    )

    workflow = RetrievalEvaluationWorkflow(
        evaluator=evaluator,
        policy=policy,
        retrieval_artifact=definition.build_retrieval_artifact(),
    )

    result = await workflow.run(definition.build_dataset())

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
    assert result.lineage.embedding_identity == definition.build_embedding_identity()

    assert result.lineage.retrieval_artifact is not None
    assert result.lineage.retrieval_artifact.retriever_type == "SemanticRetriever"
    assert result.lineage.retrieval_artifact.vector_store_type == expected_vector_store_type

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
    assert lineage["retriever_type"] == "SemanticRetriever"
    assert lineage["vector_store_type"] == expected_vector_store_type


@pytest.mark.asyncio
async def test_vehicle_quality_benchmark_workflow_preserves_hybrid_lineage() -> None:
    from rag.evaluation.dataset_registry import (
        VehicleRetrievalQualityEvaluationDatasetDefinition,
    )

    definition = VehicleRetrievalQualityEvaluationDatasetDefinition()

    retriever = await definition.build_retriever()

    evaluator = RetrievalEvaluator(
        retriever,
        k=5,
        embedding_identity=definition.build_embedding_identity(),
    )

    policy = RetrievalEvaluationPolicy(
        name="vehicle-retrieval-quality-v1",
        min_recall_at_k=1.0,
        min_precision_at_k=0.28,
        min_mrr=0.96,
        min_ndcg_at_k=0.96,
    )

    workflow = RetrievalEvaluationWorkflow(
        evaluator=evaluator,
        policy=policy,
        retrieval_artifact=definition.build_retrieval_artifact(),
    )

    result = await workflow.run(definition.build_dataset())

    assert result.passed is True
    assert result.evaluation.evaluated_queries == 16
    assert result.evaluation.successful_queries == 16
    assert result.evaluation.failed_queries == 0

    assert result.evaluation.recall_at_k == pytest.approx(1.0)
    assert result.evaluation.precision_at_k == pytest.approx(0.2875)
    assert result.evaluation.mrr == pytest.approx(0.96875)
    assert result.evaluation.ndcg_at_k == pytest.approx(0.9667784482, abs=1e-9)

    assert result.lineage.dataset_name == "vehicle-retrieval-quality"
    assert result.lineage.dataset_version == "v1"
    assert result.lineage.evaluation_policy_name == "vehicle-retrieval-quality-v1"
    assert result.lineage.evaluator_k == 5
    assert result.lineage.min_relevance_score is None
    assert result.lineage.embedding_identity == definition.build_embedding_identity()

    assert result.lineage.retrieval_artifact is not None
    assert result.lineage.retrieval_artifact.retriever_type == "HybridRetriever"
    assert result.lineage.retrieval_artifact.vector_store_type == "InMemoryVectorStore"

    hybrid_configuration = result.lineage.retrieval_artifact.hybrid_configuration
    assert hybrid_configuration is not None
    assert hybrid_configuration.candidate_k == 5
    assert hybrid_configuration.rrf_k == 60
    assert hybrid_configuration.semantic_weight == 1.0
    assert hybrid_configuration.lexical_weight == 0.5

    lineage = result.as_dict()["lineage"]

    assert lineage["dataset_name"] == "vehicle-retrieval-quality"
    assert lineage["dataset_version"] == "v1"
    assert lineage["evaluation_policy_name"] == "vehicle-retrieval-quality-v1"
    assert lineage["evaluator_k"] == 5
    assert lineage["embedding_requested_provider"] == "test-provider"
    assert lineage["embedding_requested_model"] == "vehicle-quality-benchmark"
    assert lineage["embedding_resolved_provider"] == "test-provider"
    assert lineage["embedding_resolved_model"] == "vehicle-quality-benchmark"
    assert lineage["embedding_dimension"] == 20
    assert lineage["retriever_type"] == "HybridRetriever"
    assert lineage["vector_store_type"] == "InMemoryVectorStore"
    assert lineage["hybrid_configuration"] == {
        "candidate_k": 5,
        "rrf_k": 60,
        "semantic_weight": 1.0,
        "lexical_weight": 0.5,
    }
