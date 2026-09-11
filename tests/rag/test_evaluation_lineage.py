import pytest

from rag.evaluation.lineage import RetrievalEvaluationLineage
from rag.models import EmbeddingIdentity

EMBEDDING_IDENTITY = EmbeddingIdentity(
    requested_provider="openai",
    requested_model="logical-embedding-model",
    resolved_provider="openai",
    resolved_model="text-embedding-3-small",
    dimension=1536,
)


def test_lineage_captures_evaluation_configuration() -> None:
    lineage = RetrievalEvaluationLineage(
        dataset_name="vehicle-retrieval",
        dataset_version="v2",
        evaluation_policy_name="vehicle-quality-v1",
        evaluator_k=3,
        min_relevance_score=0.75,
        embedding_identity=EMBEDDING_IDENTITY,
    )

    assert lineage.dataset_name == "vehicle-retrieval"
    assert lineage.dataset_version == "v2"
    assert lineage.evaluation_policy_name == "vehicle-quality-v1"
    assert lineage.evaluator_k == 3
    assert lineage.min_relevance_score == 0.75
    assert lineage.embedding_identity == EMBEDDING_IDENTITY


def test_lineage_as_dict_contains_embedding_provenance() -> None:
    lineage = RetrievalEvaluationLineage(
        dataset_name="vehicle-retrieval",
        dataset_version="v2",
        evaluation_policy_name="vehicle-quality-v1",
        evaluator_k=3,
        min_relevance_score=0.75,
        embedding_identity=EMBEDDING_IDENTITY,
    )

    payload = lineage.as_dict()

    assert payload == {
        "dataset_name": "vehicle-retrieval",
        "dataset_version": "v2",
        "evaluation_policy_name": "vehicle-quality-v1",
        "evaluator_k": 3,
        "min_relevance_score": 0.75,
        "embedding_requested_provider": "openai",
        "embedding_requested_model": "logical-embedding-model",
        "embedding_resolved_provider": "openai",
        "embedding_resolved_model": "text-embedding-3-small",
        "embedding_dimension": 1536,
    }


def test_lineage_allows_missing_embedding_identity() -> None:
    lineage = RetrievalEvaluationLineage(
        dataset_name="vehicle-retrieval",
        dataset_version="v1",
        evaluation_policy_name=None,
        evaluator_k=5,
        min_relevance_score=None,
    )

    payload = lineage.as_dict()

    assert payload["embedding_requested_provider"] is None
    assert payload["embedding_requested_model"] is None
    assert payload["embedding_resolved_provider"] is None
    assert payload["embedding_resolved_model"] is None
    assert payload["embedding_dimension"] is None


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("dataset_name", "   ", "dataset_name must not be empty"),
        ("dataset_version", "   ", "dataset_version must not be empty"),
        (
            "evaluation_policy_name",
            "   ",
            "evaluation_policy_name must not be empty",
        ),
    ],
)
def test_lineage_rejects_empty_identity_fields(
    field: str,
    value: str,
    message: str,
) -> None:
    kwargs = {
        "dataset_name": "dataset",
        "dataset_version": "v1",
        "evaluation_policy_name": "policy",
        "evaluator_k": 5,
        "min_relevance_score": None,
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=message):
        RetrievalEvaluationLineage(**kwargs)


def test_lineage_rejects_invalid_k() -> None:
    with pytest.raises(ValueError, match="evaluator_k must be greater than zero"):
        RetrievalEvaluationLineage(
            dataset_name="dataset",
            dataset_version="v1",
            evaluation_policy_name=None,
            evaluator_k=0,
            min_relevance_score=None,
        )


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_lineage_rejects_invalid_relevance_threshold(threshold: float) -> None:
    with pytest.raises(
        ValueError,
        match="min_relevance_score must be between 0.0 and 1.0",
    ):
        RetrievalEvaluationLineage(
            dataset_name="dataset",
            dataset_version="v1",
            evaluation_policy_name=None,
            evaluator_k=5,
            min_relevance_score=threshold,
        )
