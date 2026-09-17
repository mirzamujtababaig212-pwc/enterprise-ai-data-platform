import pytest

from rag.evaluation.datasets.enterprise_policy import (
    ENTERPRISE_POLICY_BENCHMARK_ITEMS,
    ENTERPRISE_POLICY_EMBEDDING_IDENTITY,
    ENTERPRISE_POLICY_EVALUATION_CASES,
    EnterprisePolicyBenchmarkEmbeddingService,
    enterprise_policy_benchmark_chunks,
    enterprise_policy_evaluation_cases,
)
from rag.evaluation.dataset_registry import (
    EnterprisePolicyRetrievalEvaluationDatasetDefinition,
)
from rag.evaluation.lineage import (
    HybridRetrievalConfiguration,
    RetrievalEvaluationArtifact,
)
from rag.evaluation.evaluator import RetrievalEvaluator


def test_enterprise_policy_benchmark_has_expected_items() -> None:
    assert len(ENTERPRISE_POLICY_BENCHMARK_ITEMS) == 22

    chunk_ids = {item.chunk.id for item in ENTERPRISE_POLICY_BENCHMARK_ITEMS}

    assert len(chunk_ids) == 22
    assert "policy-vendor-payment-approval" in chunk_ids
    assert "policy-access-review" in chunk_ids
    assert "policy-incident-escalation" in chunk_ids
    assert "policy-expense-threshold" in chunk_ids
    assert "policy-emergency-change" in chunk_ids
    assert "policy-restricted-data" in chunk_ids


def test_enterprise_policy_cases_reference_known_chunks() -> None:
    known_ids = {item.chunk.id for item in ENTERPRISE_POLICY_BENCHMARK_ITEMS}

    cases = enterprise_policy_evaluation_cases()

    assert len(cases) == 20
    assert cases == ENTERPRISE_POLICY_EVALUATION_CASES

    for case in cases:
        assert set(case.relevant_chunk_ids) <= known_ids
        assert case.relevance_grades is not None


def test_enterprise_policy_chunks_preserve_embedding_identity() -> None:
    chunks = enterprise_policy_benchmark_chunks()

    assert len(chunks) == len(ENTERPRISE_POLICY_BENCHMARK_ITEMS)
    assert all(chunk.embedding_identity == ENTERPRISE_POLICY_EMBEDDING_IDENTITY for chunk in chunks)


@pytest.mark.asyncio
async def test_enterprise_policy_embedding_is_deterministic() -> None:
    service = EnterprisePolicyBenchmarkEmbeddingService()

    first = await service.embed_with_metadata(
        "What approval is required before paying a new supplier?"
    )
    second = await service.embed_with_metadata(
        "What approval is required before paying a new supplier?"
    )

    assert first == second
    assert first.identity == ENTERPRISE_POLICY_EMBEDDING_IDENTITY


@pytest.mark.asyncio
async def test_enterprise_policy_definition_builds_hybrid_retriever() -> None:
    definition = EnterprisePolicyRetrievalEvaluationDatasetDefinition()

    retriever = await definition.build_retriever()

    results = await retriever.retrieve(
        "What approval is required before paying a new supplier?",
        top_k=5,
    )

    assert results
    assert len(results) <= 5
    assert results[0].chunk.content


def test_enterprise_policy_definition_builds_dataset() -> None:
    definition = EnterprisePolicyRetrievalEvaluationDatasetDefinition()

    dataset = definition.build_dataset()

    assert dataset.name == "enterprise-policy-retrieval"
    assert dataset.version == "v1"
    assert dataset.size == 20


def test_enterprise_policy_definition_preserves_embedding_identity() -> None:
    definition = EnterprisePolicyRetrievalEvaluationDatasetDefinition()

    assert definition.build_embedding_identity() == ENTERPRISE_POLICY_EMBEDDING_IDENTITY


def test_enterprise_policy_definition_builds_retrieval_artifact() -> None:
    definition = EnterprisePolicyRetrievalEvaluationDatasetDefinition()

    assert definition.build_retrieval_artifact() == RetrievalEvaluationArtifact(
        retriever_type="HybridRetriever",
        vector_store_type="InMemoryVectorStore",
        hybrid_configuration=HybridRetrievalConfiguration(
            candidate_k=5,
            rrf_k=60,
            semantic_weight=1.0,
            lexical_weight=0.5,
        ),
    )


@pytest.mark.asyncio
async def test_enterprise_policy_hybrid_retrieval_quality() -> None:
    definition = EnterprisePolicyRetrievalEvaluationDatasetDefinition()

    dataset = definition.build_dataset()
    retriever = await definition.build_retriever()

    result = await RetrievalEvaluator(
        retriever=retriever,
        k=5,
    ).evaluate(dataset.cases)

    assert result.evaluated_queries == 20
    assert result.successful_queries == 20
    assert result.failed_queries == 0

    assert result.recall_at_k == pytest.approx(0.975)
    assert result.precision_at_k == pytest.approx(0.45)
    assert result.mrr == pytest.approx(1.0)
    assert result.ndcg_at_k == pytest.approx(0.9380243335519716)

    assert result.mean_latency_ms >= 0
