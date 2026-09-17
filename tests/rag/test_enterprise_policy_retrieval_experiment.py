from datetime import datetime, timezone

import pytest

from rag.evaluation.comparison import RetrievalEvaluationExperimentComparator
from rag.evaluation.dataset_registry import (
    EnterprisePolicyRetrievalEvaluationDatasetDefinition,
)
from rag.evaluation.datasets.enterprise_policy import (
    enterprise_policy_benchmark_chunks,
)
from rag.evaluation.evaluator import RetrievalEvaluator
from rag.evaluation.lineage import RetrievalEvaluationArtifact
from rag.evaluation.policy import RetrievalEvaluationPolicy
from rag.evaluation.run import RetrievalEvaluationRun
from rag.evaluation.workflow import RetrievalEvaluationWorkflow
from rag.retrieval import HybridRetriever, SemanticRetriever
from rag.retrieval.lexical import InMemoryLexicalRetriever
from rag.stores import VectorStoreFactory

EXPERIMENT_POLICY = RetrievalEvaluationPolicy(
    name="enterprise-policy-retrieval-v1-experiment",
    min_recall_at_k=0.0,
    min_precision_at_k=0.0,
    min_mrr=0.0,
    min_ndcg_at_k=0.0,
)


def _workflow(
    retriever,
    *,
    definition,
    retrieval_artifact,
) -> RetrievalEvaluationWorkflow:
    evaluator = RetrievalEvaluator(
        retriever,
        k=5,
        embedding_identity=definition.build_embedding_identity(),
    )

    return RetrievalEvaluationWorkflow(
        evaluator=evaluator,
        policy=EXPERIMENT_POLICY,
        retrieval_artifact=retrieval_artifact,
    )


def _run(result, *, run_id: str) -> RetrievalEvaluationRun:
    return result.to_run(
        run_id=run_id,
        created_at=datetime.now(timezone.utc),
    )


async def _build_hybrid_retriever(definition):
    vector_store = VectorStoreFactory.create(backend="in_memory")

    await vector_store.upsert(enterprise_policy_benchmark_chunks())

    semantic_retriever = SemanticRetriever(
        embedding_service=definition.build_embedding_service(),
        vector_store=vector_store,
    )

    lexical_retriever = InMemoryLexicalRetriever(
        chunks=tuple(
            embedded_chunk.chunk for embedded_chunk in enterprise_policy_benchmark_chunks()
        ),
    )

    return HybridRetriever(
        semantic_retriever=semantic_retriever,
        lexical_retriever=lexical_retriever,
        candidate_k=5,
        rrf_k=60,
        semantic_weight=1.0,
        lexical_weight=0.5,
    )


async def _build_semantic_retriever(definition):
    vector_store = VectorStoreFactory.create(backend="in_memory")

    await vector_store.upsert(enterprise_policy_benchmark_chunks())

    return SemanticRetriever(
        embedding_service=definition.build_embedding_service(),
        vector_store=vector_store,
    )


@pytest.mark.asyncio
async def test_enterprise_policy_hybrid_vs_semantic_experiment():
    definition = EnterprisePolicyRetrievalEvaluationDatasetDefinition()
    dataset = definition.build_dataset()

    hybrid_retriever = await _build_hybrid_retriever(definition)
    semantic_retriever = await _build_semantic_retriever(definition)

    hybrid_workflow = _workflow(
        hybrid_retriever,
        definition=definition,
        retrieval_artifact=RetrievalEvaluationArtifact(
            retriever_type="HybridRetriever",
            vector_store_type="InMemoryVectorStore",
        ),
    )

    semantic_workflow = _workflow(
        semantic_retriever,
        definition=definition,
        retrieval_artifact=RetrievalEvaluationArtifact(
            retriever_type="SemanticRetriever",
            vector_store_type="InMemoryVectorStore",
        ),
    )

    hybrid_result = await hybrid_workflow.run(dataset)
    semantic_result = await semantic_workflow.run(dataset)

    baseline_run = _run(
        hybrid_result,
        run_id="enterprise-policy-hybrid-v1",
    )

    candidate_run = _run(
        semantic_result,
        run_id="enterprise-policy-semantic-v1",
    )

    comparison = RetrievalEvaluationExperimentComparator.compare(
        baseline_run,
        candidate_run,
    )

    assert hybrid_result.evaluation.recall_at_k == pytest.approx(0.975)
    assert hybrid_result.evaluation.precision_at_k == pytest.approx(0.45)
    assert hybrid_result.evaluation.mrr == pytest.approx(1.0)
    assert hybrid_result.evaluation.ndcg_at_k == pytest.approx(0.9380243335519716)

    assert semantic_result.evaluation.recall_at_k == pytest.approx(0.975)
    assert semantic_result.evaluation.precision_at_k == pytest.approx(0.45)
    assert semantic_result.evaluation.mrr == pytest.approx(0.975)
    assert semantic_result.evaluation.ndcg_at_k == pytest.approx(0.9069316368128588)

    assert comparison.metrics["recall_at_k"].status.value == "unchanged"
    assert comparison.metrics["precision_at_k"].status.value == "unchanged"
    assert comparison.metrics["mrr"].status.value == "regressed"
    assert comparison.metrics["ndcg_at_k"].status.value == "regressed"

    assert comparison.metrics["mrr"].delta == pytest.approx(-0.025)

    assert comparison.metrics["ndcg_at_k"].delta == pytest.approx(
        -0.031092697,
        abs=1e-6,
    )

    assert (
        comparison.metrics["mean_latency_ms"].candidate
        < comparison.metrics["mean_latency_ms"].baseline
    )
