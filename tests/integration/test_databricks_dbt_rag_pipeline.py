from unittest.mock import Mock

import pytest

from common.builders.reader_builder import ReaderBuilder
from common.readers.databricks_reader import DatabricksReader
from rag.chunking.recursive import RecursiveChunker
from rag.dbt import DbtModel, DbtModelDocumentLoader
from rag.embeddings.gateway import GatewayEmbeddingService
from rag.indexing import RAGIndexer
from rag.ingestion import RAGIngestionService
from rag.retrieval.retriever import SemanticRetriever
from rag.stores.in_memory import InMemoryVectorStore
from common.databricks.metadata import DatabricksTableMetadata
from common.provenance import databricks_source_ref


@pytest.mark.asyncio
async def test_databricks_dbt_model_flows_through_rag_pipeline():
    model = DbtModel(
        unique_id="model.vehicle_dbt.rpt_vehicle_summary",
        name="rpt_vehicle_summary",
        resource_type="model",
        database="vehicle_platform",
        schema="analytics",
        alias="rpt_vehicle_summary",
        materialization="table",
    )

    table_metadata = DatabricksTableMetadata(
        full_name="vehicle_platform.analytics.rpt_vehicle_summary",
        catalog="vehicle_platform",
        schema="analytics",
        name="rpt_vehicle_summary",
        table_type="MANAGED",
        table_id="table-123",
    )

    source_ref = databricks_source_ref(table_metadata)
    reader = ReaderBuilder.build(
        {
            "reader": {
                "type": "databricks",
                "table": "vehicle_platform.analytics.rpt_vehicle_summary",
            }
        }
    )

    assert isinstance(reader, DatabricksReader)
    assert reader.table == "vehicle_platform.analytics.rpt_vehicle_summary"

    dataframe = Mock()
    dataframe.toLocalIterator.return_value = iter(
        [
            Mock(
                asDict=Mock(
                    return_value={
                        "battery_health": "GOOD",
                        "fuel_health": "GOOD",
                        "vehicle_count": 25,
                        "avg_speed": 48.17,
                    }
                )
            )
        ]
    )

    spark = Mock()
    spark.table.return_value = dataframe

    documents = DbtModelDocumentLoader(
        model=model,
        reader=reader,
        id_fn=lambda row: (f"{model.name}:{row['battery_health']}:{row['fuel_health']}"),
        content_fn=lambda row: (
            f"Databricks fleet summary: "
            f"{row['vehicle_count']} vehicles with "
            f"average speed {row['avg_speed']} and "
            f"battery health {row['battery_health']} "
            f"and fuel health {row['fuel_health']}."
        ),
        metadata_fn=lambda row: {
            "source": "databricks",
            "dataset": model.name,
            "battery_health": row["battery_health"],
            "fuel_health": row["fuel_health"],
            "source_ref": source_ref.to_dict(),
        },
    )

    documents = documents.load(spark)

    spark.table.assert_called_once_with(
        "vehicle_platform.analytics.rpt_vehicle_summary",
    )

    assert len(documents) == 1
    assert documents[0].id == "rpt_vehicle_summary:GOOD:GOOD"
    assert documents[0].metadata["source"] == "databricks"
    assert documents[0].metadata["dbt_model"] == "rpt_vehicle_summary"
    assert documents[0].metadata["dbt_database"] == "vehicle_platform"
    assert documents[0].metadata["dbt_schema"] == "analytics"
    assert documents[0].metadata["source_ref"] == {
        "platform": "databricks",
        "object_type": "table",
        "object_name": "rpt_vehicle_summary",
        "object_id": "table-123",
        "namespace": "vehicle_platform.analytics",
    }

    vector_store = InMemoryVectorStore()

    embedding_service = GatewayEmbeddingService(
        provider="mock",
        model="mock-embedding",
    )

    indexer = RAGIndexer(
        chunker=RecursiveChunker(),
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    ingestion_service = RAGIngestionService(
        indexer=indexer,
    )

    result = await ingestion_service.ingest(documents)

    assert result.documents_processed == 1
    assert result.documents_indexed == 1
    assert result.chunks_created >= 1

    retriever = SemanticRetriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    results = await retriever.retrieve(
        "What is the average speed for the fleet with good battery and fuel health?",
        top_k=5,
    )

    assert results
    assert results[0].chunk.metadata["dbt_model"] == "rpt_vehicle_summary"
    assert results[0].chunk.metadata["source"] == "databricks"
    assert results[0].chunk.metadata["source_ref"] == {
        "platform": "databricks",
        "object_type": "table",
        "object_name": "rpt_vehicle_summary",
        "object_id": "table-123",
        "namespace": "vehicle_platform.analytics",
    }
