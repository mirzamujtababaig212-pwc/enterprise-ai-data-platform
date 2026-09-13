from __future__ import annotations

from datetime import datetime

import pytest

from ai_platform.llm_gateway.routing.router import Router
from common.provenance.source import EnterpriseSourceRef
from common.readers.delta_reader import DeltaReader
from rag.chunking.recursive import RecursiveChunker
from rag.embeddings.gateway import GatewayEmbeddingService
from rag.indexing import RAGIndexer
from rag.loaders import GoldVehicleMetricsDocumentLoader
from rag.retrieval import SemanticRetriever
from rag.stores import InMemoryVectorStore


@pytest.mark.asyncio
async def test_gold_vehicle_metrics_flows_into_rag_retrieval(
    spark,
    tmp_path,
) -> None:
    dataframe = spark.createDataFrame(
        [
            (
                "V001",
                3,
                48.17,
                45.5,
                51.8,
                datetime(2026, 1, 1, 10, 0, 0),
                datetime(2026, 1, 1, 12, 0, 0),
            ),
            (
                "V002",
                2,
                34.73,
                32.1,
                36.4,
                datetime(2026, 1, 2, 10, 0, 0),
                datetime(2026, 1, 2, 11, 0, 0),
            ),
        ],
        [
            "vehicle_id",
            "event_count",
            "avg_speed",
            "min_speed",
            "max_speed",
            "first_event_time",
            "last_event_time",
        ],
    )

    gold_path = tmp_path / "gold_vehicle_metrics"
    dataframe.write.format("delta").mode("overwrite").save(str(gold_path))

    reader = DeltaReader(path=str(gold_path))

    loader = GoldVehicleMetricsDocumentLoader(
        reader=reader,
    )

    documents = loader.load(spark)

    assert len(documents) == 2

    source_ref = EnterpriseSourceRef(
        platform="delta",
        object_type="table",
        object_name="gold.vehicle_metrics",
    ).to_dict()

    for document in documents:
        document.metadata["source_ref"] = source_ref

    documents_by_id = {document.id: document for document in documents}

    assert documents_by_id["vehicle:V001"].metadata["source_ref"] == {
        "platform": "delta",
        "object_type": "table",
        "object_name": "gold.vehicle_metrics",
    }

    gateway = Router()

    embedding_service = GatewayEmbeddingService(
        provider="mock",
        model="mock-embedding",
        gateway_router=gateway,
    )

    vector_store = InMemoryVectorStore()

    indexer = RAGIndexer(
        chunker=RecursiveChunker(
            chunk_size=500,
            overlap=50,
        ),
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    for document in documents:
        indexed = await indexer.index(document)
        assert indexed

    retriever = SemanticRetriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    results = await retriever.retrieve(
        "What was the average speed of vehicle V001?",
        top_k=3,
    )

    assert results
    assert results[0].chunk.document_id == "vehicle:V001"
    assert "Average speed was 48.17." in results[0].chunk.content
    assert results[0].chunk.metadata["source"] == "gold.vehicle_metrics"
    assert results[0].chunk.metadata["vehicle_id"] == "V001"
    assert results[0].chunk.metadata["source_ref"] == source_ref
