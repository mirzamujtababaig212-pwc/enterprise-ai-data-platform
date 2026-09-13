from __future__ import annotations

import os

import pytest
from qdrant_client import AsyncQdrantClient

from rag.evaluation.dataset_registry import VehicleRetrievalEvaluationDatasetDefinition
from rag.evaluation.evaluator import RetrievalEvaluator
from rag.evaluation.policy import RetrievalEvaluationPolicy
from rag.evaluation.workflow import RetrievalEvaluationWorkflow
from rag.evaluation.datasets.vehicle import vehicle_benchmark_chunks
from rag.retrieval import SemanticRetriever
from rag.stores.qdrant import QdrantVectorStore

QDRANT_TEST_URL = os.getenv("QDRANT_TEST_URL", "http://localhost:6333")
QDRANT_TEST_COLLECTION = os.getenv(
    "QDRANT_TEST_COLLECTION",
    "vehicle-retrieval-eval-integration",
)


@pytest.mark.asyncio
async def test_vehicle_benchmark_workflow_preserves_qdrant_retrieval_lineage() -> None:
    if os.getenv("RUN_QDRANT_INTEGRATION") != "1":
        pytest.skip("Set RUN_QDRANT_INTEGRATION=1 to run the Qdrant evaluation integration test")

    client = AsyncQdrantClient(
        url=QDRANT_TEST_URL,
    )

    try:
        assert await client.get_collections() is not None

        if await client.collection_exists(QDRANT_TEST_COLLECTION):
            await client.delete_collection(QDRANT_TEST_COLLECTION)

        store = QdrantVectorStore(
            client=client,
            collection_name=QDRANT_TEST_COLLECTION,
        )

        await store.upsert(vehicle_benchmark_chunks())

        definition = VehicleRetrievalEvaluationDatasetDefinition(
            vector_store_backend="qdrant",
        )

        retriever = SemanticRetriever(
            embedding_service=definition.build_embedding_service(),
            vector_store=store,
        )

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

        assert result.lineage.retrieval_artifact is not None
        assert result.lineage.retrieval_artifact.retriever_type == "SemanticRetriever"
        assert result.lineage.retrieval_artifact.vector_store_type == "QdrantVectorStore"

        lineage = result.as_dict()["lineage"]

        assert lineage["retriever_type"] == "SemanticRetriever"
        assert lineage["vector_store_type"] == "QdrantVectorStore"

    finally:
        if await client.collection_exists(QDRANT_TEST_COLLECTION):
            await client.delete_collection(QDRANT_TEST_COLLECTION)

        await client.close()
