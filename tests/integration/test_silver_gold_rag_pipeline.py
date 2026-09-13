from __future__ import annotations


import pytest

from ai_platform.llm_gateway.routing.router import Router
from common.provenance.source import EnterpriseSourceRef
from common.readers.delta_reader import DeltaReader
from common.writers.delta_writer import DeltaWriter
from rag.chunking.recursive import RecursiveChunker
from rag.embeddings.gateway import GatewayEmbeddingService
from rag.indexing import RAGIndexer
from rag.loaders import GoldVehicleMetricsDocumentLoader
from rag.retrieval import SemanticRetriever
from rag.stores import InMemoryVectorStore
from spark.transformations.silver_to_gold_transformer import (
    SilverToGoldTransformer,
)
from spark.validation.gold_validator import GoldValidator
from spark.validation.silver_gold_reconciliation import (
    SilverGoldReconciliation,
)


@pytest.mark.asyncio
async def test_silver_to_gold_to_rag_vertical_slice(
    spark,
    tmp_path,
) -> None:
    silver_df = spark.createDataFrame(
        [
            (
                "V001",
                "2026-01-01 10:00:00",
                45.5,
            ),
            (
                "V001",
                "2026-01-01 11:00:00",
                48.0,
            ),
            (
                "V001",
                "2026-01-01 12:00:00",
                50.01,
            ),
            (
                "V002",
                "2026-01-02 10:00:00",
                32.1,
            ),
            (
                "V002",
                "2026-01-02 11:00:00",
                36.4,
            ),
            (
                None,
                "2026-01-02 12:00:00",
                99.0,
            ),
            (
                "V003",
                "2026-01-03 10:00:00",
                -5.0,
            ),
        ],
        [
            "vehicle_id",
            "event_time",
            "speed",
        ],
    )

    # ------------------------------------------------------------------
    # Silver -> Gold
    # ------------------------------------------------------------------

    gold_df = SilverToGoldTransformer.transform(silver_df)

    assert gold_df.count() == 2

    GoldValidator.validate(gold_df)

    reconciliation = SilverGoldReconciliation.reconcile(
        silver_df=silver_df,
        gold_df=gold_df,
    )

    assert reconciliation["eligible_silver_rows"] == 5
    assert reconciliation["expected_gold_rows"] == 2
    assert reconciliation["actual_gold_rows"] == 2
    assert reconciliation["distinct_silver_vehicles"] == 2
    assert reconciliation["expected_event_count"] == 5
    assert reconciliation["actual_event_count"] == 5
    assert reconciliation["mismatched_rows"] == 0

    # ------------------------------------------------------------------
    # Validated Gold -> Delta
    # ------------------------------------------------------------------

    gold_path = tmp_path / "gold_vehicle_metrics"

    writer = DeltaWriter(
        table="gold.vehicle_metrics",
        path=str(gold_path),
        mode="overwrite",
    )

    writer.write(gold_df)

    # ------------------------------------------------------------------
    # Delta -> canonical RAG Documents
    # ------------------------------------------------------------------

    reader = DeltaReader(
        path=str(gold_path),
    )

    loader = GoldVehicleMetricsDocumentLoader(
        reader=reader,
    )

    documents = loader.load(spark)

    assert {document.id for document in documents} == {
        "vehicle:V001",
        "vehicle:V002",
    }

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

    assert "Average speed was 47.84." in documents_by_id["vehicle:V001"].content

    # ------------------------------------------------------------------
    # RAG indexing
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # RAG retrieval
    # ------------------------------------------------------------------

    retriever = SemanticRetriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    results = await retriever.retrieve(
        "What was the average speed of vehicle V001?",
        top_k=2,
    )

    assert results
    assert results[0].chunk.document_id == "vehicle:V001"
    assert "Average speed was 47.84." in results[0].chunk.content
    assert results[0].chunk.metadata["source"] == "gold.vehicle_metrics"
    assert results[0].chunk.metadata["data_layer"] == "gold"
    assert results[0].chunk.metadata["dataset"] == "vehicle_metrics"
    assert results[0].chunk.metadata["vehicle_id"] == "V001"
    assert results[0].chunk.metadata["source_ref"] == source_ref
