import os

import pytest
from qdrant_client import AsyncQdrantClient

from rag.evaluation.datasets.vehicle_retrieval_quality import (
    VEHICLE_QUALITY_EMBEDDING_IDENTITY,
    VehicleQualityBenchmarkEmbeddingService,
    vehicle_quality_benchmark_chunks,
    vehicle_quality_evaluation_cases,
)
from rag.evaluation.evaluator import RetrievalEvaluator
from rag.retrieval.retriever import SemanticRetriever
from rag.stores.qdrant import QdrantVectorStore

pytestmark = pytest.mark.asyncio


@pytest.mark.skipif(
    os.getenv("RUN_QDRANT_INTEGRATION") != "1",
    reason="Set RUN_QDRANT_INTEGRATION=1 to run Qdrant integration tests",
)
async def test_quality_benchmark_qdrant_regression() -> None:
    qdrant_url = os.getenv("QDRANT_TEST_URL", "http://localhost:6333")
    collection_name = "vehicle-quality-retrieval-regression"

    client = AsyncQdrantClient(url=qdrant_url)

    try:
        await client.delete_collection(collection_name=collection_name)

        chunks = vehicle_quality_benchmark_chunks()

        store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
        )
        await store.upsert(chunks)

        collection = await client.get_collection(collection_name)
        assert collection.config.params.vectors.size == 20

        points = await client.count(
            collection_name=collection_name,
            exact=True,
        )
        assert points.count == 24

        embedding_service = VehicleQualityBenchmarkEmbeddingService()
        retriever = SemanticRetriever(
            embedding_service=embedding_service,
            vector_store=store,
        )

        evaluator = RetrievalEvaluator(
            retriever,
            k=5,
            embedding_identity=VEHICLE_QUALITY_EMBEDDING_IDENTITY,
        )

        result = await evaluator.evaluate(vehicle_quality_evaluation_cases())

        assert result.evaluated_queries == 16
        assert result.successful_queries == 16
        assert result.failed_queries == 0

        assert result.recall_at_k == pytest.approx(1.0)
        assert result.precision_at_k == pytest.approx(0.2875)
        assert result.mrr == pytest.approx(1.0)
        assert result.ndcg_at_k == pytest.approx(0.988067, abs=1e-5)

    finally:
        await client.delete_collection(collection_name=collection_name)
        await client.close()
