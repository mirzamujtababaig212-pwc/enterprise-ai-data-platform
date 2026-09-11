from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime

from rag.evaluation.dataset import RetrievalEvaluationDataset
from rag.evaluation.datasets.vehicle import (
    VEHICLE_EMBEDDING_IDENTITY,
    VehicleBenchmarkEmbeddingService,
    vehicle_benchmark_chunks,
    vehicle_evaluation_cases,
)
from rag.evaluation.evaluator import RetrievalEvaluator
from rag.evaluation.policy import RetrievalEvaluationPolicy
from rag.evaluation.release import RetrievalEvaluationReleaseGate
from rag.evaluation.run import RetrievalEvaluationRun
from rag.evaluation.workflow import RetrievalEvaluationWorkflow
from rag.retrieval import SemanticRetriever
from rag.stores import InMemoryVectorStore

DEFAULT_POLICY_NAME = "vehicle-retrieval-quality-v2"


async def evaluate() -> RetrievalEvaluationRun:
    """Execute the deterministic vehicle retrieval evaluation."""
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
        name=DEFAULT_POLICY_NAME,
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

    created_at = datetime.now(UTC)
    run_id = f"ci-{dataset.name}-{dataset.version}-" f"{created_at.strftime('%Y%m%dT%H%M%SZ')}"

    return result.to_run(
        run_id=run_id,
        created_at=created_at,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the deterministic RAG release evaluation.")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the human-readable evaluation summary.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run = asyncio.run(evaluate())
    decision = RetrievalEvaluationReleaseGate.evaluate(run)

    if not args.quiet:
        evaluation = run.evaluation
        print("=== RAG Release Evaluation ===")
        print(f"Run ID:              {run.run_id}")
        print(f"Dataset:             {run.lineage.dataset_name}")
        print(f"Dataset version:     {run.lineage.dataset_version}")
        print(f"Evaluated queries:   {evaluation.evaluated_queries}")
        print(f"Successful queries:  {evaluation.successful_queries}")
        print(f"Failed queries:      {evaluation.failed_queries}")
        print(f"Recall@3:            {evaluation.recall_at_k:.4f}")
        print(f"Precision@3:         {evaluation.precision_at_k:.4f}")
        print(f"MRR:                 {evaluation.mrr:.4f}")
        print(f"NDCG@3:              {evaluation.ndcg_at_k:.4f}")
        print(f"Mean latency (ms):   {evaluation.mean_latency_ms:.4f}")
        print(f"Quality gate:        {'PASS' if run.passed else 'FAIL'}")
        print(f"Release decision:    {'PASS' if decision.passed else 'FAIL'}")

        if decision.errors:
            print("Release errors:")
            for error in decision.errors:
                print(f"  - {error}")

    return 0 if decision.passed else 1


if __name__ == "__main__":
    sys.exit(main())
