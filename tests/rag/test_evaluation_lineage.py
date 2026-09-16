import pytest

from rag.evaluation.lineage import (
    HybridRetrievalConfiguration,
    RetrievalEvaluationArtifact,
    RetrievalEvaluationLineage,
)
from rag.models import EmbeddingIdentity

TEST_EMBEDDING_IDENTITY = EmbeddingIdentity(
    requested_provider="openai",
    requested_model="logical-embedding-model",
    resolved_provider="openai",
    resolved_model="text-embedding-3-small",
    dimension=1536,
)


def test_hybrid_retrieval_configuration_captures_fusion_parameters() -> None:
    configuration = HybridRetrievalConfiguration(
        candidate_k=5,
        rrf_k=60,
        semantic_weight=1.0,
        lexical_weight=0.5,
    )

    assert configuration.candidate_k == 5
    assert configuration.rrf_k == 60
    assert configuration.semantic_weight == 1.0
    assert configuration.lexical_weight == 0.5
    assert configuration.as_dict() == {
        "candidate_k": 5,
        "rrf_k": 60,
        "semantic_weight": 1.0,
        "lexical_weight": 0.5,
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("candidate_k", 0, "candidate_k must be greater than zero"),
        ("rrf_k", 0, "rrf_k must be greater than zero"),
        ("semantic_weight", -0.1, "semantic_weight must be greater than or equal to zero"),
        ("lexical_weight", -0.1, "lexical_weight must be greater than or equal to zero"),
    ],
)
def test_hybrid_retrieval_configuration_rejects_invalid_parameters(
    field: str,
    value: float | int,
    message: str,
) -> None:
    values = {
        "candidate_k": 5,
        "rrf_k": 60,
        "semantic_weight": 1.0,
        "lexical_weight": 0.5,
    }
    values[field] = value

    with pytest.raises(ValueError, match=message):
        HybridRetrievalConfiguration(**values)


def test_hybrid_retrieval_configuration_rejects_zero_weights() -> None:
    with pytest.raises(
        ValueError,
        match="At least one retrieval weight must be greater than zero",
    ):
        HybridRetrievalConfiguration(
            candidate_k=5,
            rrf_k=60,
            semantic_weight=0.0,
            lexical_weight=0.0,
        )


def test_retrieval_artifact_preserves_hybrid_configuration() -> None:
    configuration = HybridRetrievalConfiguration(
        candidate_k=5,
        rrf_k=60,
        semantic_weight=1.0,
        lexical_weight=0.5,
    )

    artifact = RetrievalEvaluationArtifact(
        retriever_type="HybridRetriever",
        vector_store_type="InMemoryVectorStore",
        hybrid_configuration=configuration,
    )

    assert artifact.hybrid_configuration == configuration
    assert artifact.as_dict() == {
        "retriever_type": "HybridRetriever",
        "vector_store_type": "InMemoryVectorStore",
        "hybrid_configuration": {
            "candidate_k": 5,
            "rrf_k": 60,
            "semantic_weight": 1.0,
            "lexical_weight": 0.5,
        },
    }


def test_lineage_captures_evaluation_configuration() -> None:
    lineage = RetrievalEvaluationLineage(
        dataset_name="vehicle-retrieval",
        dataset_version="v2",
        evaluation_policy_name="vehicle-quality-v2",
        min_recall_at_k=1.0,
        min_precision_at_k=0.8,
        min_mrr=1.0,
        min_ndcg_at_k=0.95,
        max_mean_latency_ms=250.0,
        min_abstention_accuracy=0.9,
        evaluator_k=3,
        min_relevance_score=0.75,
        embedding_identity=TEST_EMBEDDING_IDENTITY,
        retrieval_artifact=RetrievalEvaluationArtifact(
            retriever_type="SemanticRetriever",
            vector_store_type="InMemoryVectorStore",
        ),
    )

    assert lineage.dataset_name == "vehicle-retrieval"
    assert lineage.dataset_version == "v2"
    assert lineage.evaluation_policy_name == "vehicle-quality-v2"
    assert lineage.min_recall_at_k == 1.0
    assert lineage.min_precision_at_k == 0.8
    assert lineage.min_mrr == 1.0
    assert lineage.min_ndcg_at_k == 0.95
    assert lineage.max_mean_latency_ms == 250.0
    assert lineage.min_abstention_accuracy == 0.9
    assert lineage.evaluator_k == 3
    assert lineage.min_relevance_score == 0.75
    assert lineage.embedding_identity == TEST_EMBEDDING_IDENTITY
    assert lineage.retrieval_artifact == RetrievalEvaluationArtifact(
        retriever_type="SemanticRetriever",
        vector_store_type="InMemoryVectorStore",
    )


def test_lineage_as_dict_contains_policy_snapshot_and_embedding_provenance() -> None:
    lineage = RetrievalEvaluationLineage(
        dataset_name="vehicle-retrieval",
        dataset_version="v2",
        evaluation_policy_name="vehicle-quality-v2",
        min_recall_at_k=1.0,
        min_precision_at_k=0.8,
        min_mrr=1.0,
        min_ndcg_at_k=0.95,
        max_mean_latency_ms=250.0,
        min_abstention_accuracy=0.9,
        evaluator_k=3,
        min_relevance_score=0.75,
        embedding_identity=TEST_EMBEDDING_IDENTITY,
        retrieval_artifact=RetrievalEvaluationArtifact(
            retriever_type="SemanticRetriever",
            vector_store_type="InMemoryVectorStore",
        ),
    )

    payload = lineage.as_dict()

    assert payload["dataset_name"] == "vehicle-retrieval"
    assert payload["dataset_version"] == "v2"
    assert payload["evaluation_policy_name"] == "vehicle-quality-v2"

    assert payload["min_recall_at_k"] == 1.0
    assert payload["min_precision_at_k"] == 0.8
    assert payload["min_mrr"] == 1.0
    assert payload["min_ndcg_at_k"] == 0.95
    assert payload["max_mean_latency_ms"] == 250.0
    assert payload["min_abstention_accuracy"] == 0.9

    assert payload["evaluator_k"] == 3
    assert payload["min_relevance_score"] == 0.75

    assert payload["embedding_requested_provider"] == "openai"
    assert payload["embedding_requested_model"] == "logical-embedding-model"
    assert payload["embedding_resolved_provider"] == "openai"
    assert payload["embedding_resolved_model"] == "text-embedding-3-small"
    assert payload["embedding_dimension"] == 1536
    assert payload["retriever_type"] == "SemanticRetriever"
    assert payload["vector_store_type"] == "InMemoryVectorStore"
    assert payload["hybrid_configuration"] is None


def test_lineage_allows_missing_embedding_identity() -> None:
    lineage = RetrievalEvaluationLineage(
        dataset_name="vehicle-retrieval",
        dataset_version="v1",
        evaluation_policy_name="vehicle-quality-v1",
        min_recall_at_k=0.8,
        min_precision_at_k=None,
        min_mrr=None,
        min_ndcg_at_k=None,
        max_mean_latency_ms=None,
        min_abstention_accuracy=None,
        evaluator_k=5,
        min_relevance_score=None,
    )

    payload = lineage.as_dict()

    assert lineage.embedding_identity is None
    assert payload["embedding_requested_provider"] is None
    assert payload["embedding_requested_model"] is None
    assert payload["embedding_resolved_provider"] is None
    assert payload["embedding_resolved_model"] is None
    assert payload["embedding_dimension"] is None


@pytest.mark.parametrize(
    "field",
    [
        "min_recall_at_k",
        "min_precision_at_k",
        "min_mrr",
        "min_ndcg_at_k",
        "min_abstention_accuracy",
    ],
)
def test_lineage_rejects_invalid_quality_threshold(field: str) -> None:
    kwargs = {
        "dataset_name": "vehicle-retrieval",
        "dataset_version": "v1",
        "evaluation_policy_name": "vehicle-quality-v1",
        "min_recall_at_k": None,
        "min_precision_at_k": None,
        "min_mrr": None,
        "min_ndcg_at_k": None,
        "max_mean_latency_ms": None,
        "min_abstention_accuracy": None,
        "evaluator_k": 5,
        "min_relevance_score": None,
    }
    kwargs[field] = 1.1

    with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
        RetrievalEvaluationLineage(**kwargs)


def test_lineage_rejects_negative_latency_threshold() -> None:
    with pytest.raises(ValueError, match="max_mean_latency_ms must be non-negative"):
        RetrievalEvaluationLineage(
            dataset_name="vehicle-retrieval",
            dataset_version="v1",
            evaluation_policy_name="vehicle-quality-v1",
            min_recall_at_k=0.8,
            min_precision_at_k=None,
            min_mrr=None,
            min_ndcg_at_k=None,
            max_mean_latency_ms=-1.0,
            min_abstention_accuracy=None,
            evaluator_k=5,
            min_relevance_score=None,
        )


def test_lineage_rejects_empty_identity_fields() -> None:
    with pytest.raises(ValueError, match="dataset_name"):
        RetrievalEvaluationLineage(
            dataset_name=" ",
            dataset_version="v1",
            evaluation_policy_name=None,
            min_recall_at_k=0.8,
            min_precision_at_k=None,
            min_mrr=None,
            min_ndcg_at_k=None,
            max_mean_latency_ms=None,
            min_abstention_accuracy=None,
            evaluator_k=5,
            min_relevance_score=None,
        )


def test_lineage_rejects_invalid_k() -> None:
    with pytest.raises(ValueError, match="evaluator_k must be greater than zero"):
        RetrievalEvaluationLineage(
            dataset_name="vehicle-retrieval",
            dataset_version="v1",
            evaluation_policy_name=None,
            min_recall_at_k=0.8,
            min_precision_at_k=None,
            min_mrr=None,
            min_ndcg_at_k=None,
            max_mean_latency_ms=None,
            min_abstention_accuracy=None,
            evaluator_k=0,
            min_relevance_score=None,
        )


def test_lineage_rejects_invalid_relevance_threshold() -> None:
    with pytest.raises(
        ValueError,
        match="min_relevance_score must be between 0.0 and 1.0",
    ):
        RetrievalEvaluationLineage(
            dataset_name="vehicle-retrieval",
            dataset_version="v1",
            evaluation_policy_name=None,
            min_recall_at_k=0.8,
            min_precision_at_k=None,
            min_mrr=None,
            min_ndcg_at_k=None,
            max_mean_latency_ms=None,
            min_abstention_accuracy=None,
            evaluator_k=5,
            min_relevance_score=1.1,
        )


@pytest.mark.parametrize(
    "field",
    ["retriever_type", "vector_store_type"],
)
def test_retrieval_evaluation_artifact_rejects_empty_fields(field: str) -> None:
    kwargs = {
        "retriever_type": "SemanticRetriever",
        "vector_store_type": "InMemoryVectorStore",
    }
    kwargs[field] = " "

    with pytest.raises(ValueError, match=f"{field} must not be empty"):
        RetrievalEvaluationArtifact(**kwargs)
