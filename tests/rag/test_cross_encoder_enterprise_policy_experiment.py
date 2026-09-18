from datetime import datetime, timezone

import pytest

from rag.evaluation.comparison import (
    RetrievalEvaluationExperimentComparator,
)
from rag.evaluation.dataset_registry import (
    EnterprisePolicyRetrievalEvaluationDatasetDefinition,
)
from rag.evaluation.evaluator import RetrievalEvaluator
from rag.evaluation.lineage import RerankerConfiguration
from rag.evaluation.policy import RetrievalEvaluationPolicy
from rag.evaluation.run import RetrievalEvaluationRun
from rag.evaluation.workflow import RetrievalEvaluationWorkflow
from rag.retrieval import CrossEncoderReranker, RerankingRetriever

EXPERIMENT_POLICY = RetrievalEvaluationPolicy(
    name="enterprise-policy-retrieval-v1-cross-encoder-experiment",
    min_recall_at_k=0.0,
    min_precision_at_k=0.0,
    min_mrr=0.0,
    min_ndcg_at_k=0.0,
)


def _workflow(
    retriever,
    *,
    definition: EnterprisePolicyRetrievalEvaluationDatasetDefinition,
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


def _run(
    result,
    *,
    run_id: str,
) -> RetrievalEvaluationRun:
    return result.to_run(
        run_id=run_id,
        created_at=datetime.now(timezone.utc),
    )


@pytest.mark.asyncio
async def test_cross_encoder_enterprise_policy_experiment() -> None:
    definition = EnterprisePolicyRetrievalEvaluationDatasetDefinition()
    dataset = definition.build_dataset()

    baseline_retriever = await definition.build_retriever()

    baseline_workflow = _workflow(
        baseline_retriever,
        definition=definition,
        retrieval_artifact=definition.build_retrieval_artifact(),
    )
    baseline_result = await baseline_workflow.run(dataset)
    baseline_run = _run(
        baseline_result,
        run_id="enterprise-policy-hybrid-baseline",
    )

    reranker = CrossEncoderReranker()
    candidate_retriever = RerankingRetriever(
        retriever=baseline_retriever,
        reranker=reranker,
        candidate_k=5,
    )

    baseline_artifact = definition.build_retrieval_artifact()
    candidate_artifact = type(baseline_artifact)(
        retriever_type="RerankingRetriever",
        vector_store_type=baseline_artifact.vector_store_type,
        hybrid_configuration=baseline_artifact.hybrid_configuration,
        reranker_configuration=RerankerConfiguration(
            type="cross_encoder",
            model_id=reranker.model_id,
            onnx_filename=reranker.onnx_filename,
            max_length=reranker.max_length,
            candidate_k=5,
        ),
    )

    candidate_workflow = _workflow(
        candidate_retriever,
        definition=definition,
        retrieval_artifact=candidate_artifact,
    )
    candidate_result = await candidate_workflow.run(dataset)
    candidate_run = _run(
        candidate_result,
        run_id="enterprise-policy-cross-encoder-candidate",
    )

    comparison = RetrievalEvaluationExperimentComparator.compare(
        baseline_run,
        candidate_run,
    )

    assert baseline_run.passed is True
    assert candidate_run.passed is True

    assert baseline_run.evaluation.evaluated_queries == 20
    assert candidate_run.evaluation.evaluated_queries == 20

    assert baseline_run.evaluation.failed_queries == 0
    assert candidate_run.evaluation.failed_queries == 0

    assert baseline_run.evaluation.recall_at_k >= 0.0
    assert baseline_run.evaluation.precision_at_k >= 0.0
    assert baseline_run.evaluation.mrr >= 0.0
    assert baseline_run.evaluation.ndcg_at_k >= 0.0

    assert candidate_run.evaluation.recall_at_k >= 0.0
    assert candidate_run.evaluation.precision_at_k >= 0.0
    assert candidate_run.evaluation.mrr >= 0.0
    assert candidate_run.evaluation.ndcg_at_k >= 0.0

    assert baseline_run.lineage.retrieval_artifact is not None
    assert baseline_run.lineage.retrieval_artifact.retriever_type == "HybridRetriever"

    assert candidate_run.lineage.retrieval_artifact is not None
    assert candidate_run.lineage.retrieval_artifact.retriever_type == "RerankingRetriever"

    candidate_artifact = candidate_run.lineage.retrieval_artifact
    assert candidate_artifact.hybrid_configuration is not None
    assert candidate_artifact.hybrid_configuration.candidate_k == 5
    assert candidate_artifact.hybrid_configuration.rrf_k == 60
    assert candidate_artifact.hybrid_configuration.semantic_weight == 1.0
    assert candidate_artifact.hybrid_configuration.lexical_weight == 0.5

    assert candidate_artifact.reranker_configuration == RerankerConfiguration(
        type="cross_encoder",
        model_id=reranker.model_id,
        onnx_filename=reranker.onnx_filename,
        max_length=reranker.max_length,
        candidate_k=5,
    )

    assert set(comparison.metrics) >= {
        "recall_at_k",
        "precision_at_k",
        "mrr",
        "ndcg_at_k",
        "mean_latency_ms",
    }
