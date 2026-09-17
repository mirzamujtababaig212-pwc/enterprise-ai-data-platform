from datetime import datetime, timezone

import pytest

from rag.evaluation.comparison import (
    RetrievalEvaluationExperimentComparator,
)
from rag.evaluation.dataset_registry import (
    VehicleRetrievalQualityEvaluationDatasetDefinition,
)
from rag.evaluation.evaluator import RetrievalEvaluator
from rag.evaluation.policy import RetrievalEvaluationPolicy
from rag.evaluation.run import RetrievalEvaluationRun
from rag.evaluation.workflow import RetrievalEvaluationWorkflow
from rag.retrieval import CrossEncoderReranker, RerankingRetriever

EXPERIMENT_POLICY = RetrievalEvaluationPolicy(
    name="vehicle-retrieval-quality-v1-experiment",
    min_recall_at_k=1.0,
    min_precision_at_k=0.28,
    min_mrr=0.96,
    min_ndcg_at_k=0.96,
)


def _workflow(
    retriever,
    *,
    definition: VehicleRetrievalQualityEvaluationDatasetDefinition,
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
async def test_cross_encoder_vehicle_quality_experiment() -> None:
    definition = VehicleRetrievalQualityEvaluationDatasetDefinition()
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
        run_id="vehicle-quality-hybrid-baseline",
    )

    reranker = CrossEncoderReranker()
    candidate_retriever = RerankingRetriever(
        retriever=baseline_retriever,
        reranker=reranker,
        candidate_k=5,
    )

    candidate_artifact = definition.build_retrieval_artifact()
    candidate_artifact = type(candidate_artifact)(
        retriever_type="RerankingRetriever",
        vector_store_type=candidate_artifact.vector_store_type,
    )

    candidate_workflow = _workflow(
        candidate_retriever,
        definition=definition,
        retrieval_artifact=candidate_artifact,
    )
    candidate_result = await candidate_workflow.run(dataset)
    candidate_run = _run(
        candidate_result,
        run_id="vehicle-quality-cross-encoder-candidate",
    )

    comparison = RetrievalEvaluationExperimentComparator.compare(
        baseline_run,
        candidate_run,
    )

    assert baseline_run.passed is True
    assert candidate_run.passed is True

    assert baseline_run.evaluation.recall_at_k == pytest.approx(1.0)
    assert baseline_run.evaluation.precision_at_k == pytest.approx(0.2875)
    assert baseline_run.evaluation.mrr == pytest.approx(0.96875)
    assert baseline_run.evaluation.ndcg_at_k == pytest.approx(
        0.9667784482,
        abs=1e-9,
    )

    assert candidate_run.evaluation.recall_at_k == pytest.approx(1.0)
    assert candidate_run.evaluation.precision_at_k == pytest.approx(0.2875)
    assert candidate_run.evaluation.mrr == pytest.approx(1.0)
    assert candidate_run.evaluation.ndcg_at_k == pytest.approx(
        0.964524,
        abs=1e-6,
    )

    assert baseline_run.lineage.retrieval_artifact is not None
    assert baseline_run.lineage.retrieval_artifact.retriever_type == "HybridRetriever"

    assert candidate_run.lineage.retrieval_artifact is not None
    assert candidate_run.lineage.retrieval_artifact.retriever_type == "RerankingRetriever"

    assert comparison.metrics["mrr"].status.value == "improved"
    assert comparison.metrics["mrr"].delta == pytest.approx(0.03125)

    assert comparison.metrics["ndcg_at_k"].status.value == "regressed"
    assert comparison.metrics["ndcg_at_k"].delta == pytest.approx(
        -0.002254,
        abs=1e-6,
    )

    assert comparison.metrics["recall_at_k"].status.value == "unchanged"
    assert comparison.metrics["precision_at_k"].status.value == "unchanged"
