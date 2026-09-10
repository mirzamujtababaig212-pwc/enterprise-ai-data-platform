from unittest.mock import Mock, patch

import pytest

from common.builders.reader_builder import ReaderBuilder
from common.readers.snowflake_reader import SnowflakeReader
from rag.dbt import DbtModel, DbtModelDocumentLoader
from rag.embeddings.gateway import GatewayEmbeddingService
from rag.indexing import RAGIndexer
from rag.ingestion import RAGIngestionService
from rag.retrieval.retriever import SemanticRetriever
from rag.stores.in_memory import InMemoryVectorStore
from rag.chunking.recursive import RecursiveChunker


@pytest.mark.asyncio
async def test_snowflake_dbt_model_flows_through_rag_pipeline():
    model = DbtModel(
        unique_id="model.vehicle_dbt.rpt_vehicle_summary",
        name="rpt_vehicle_summary",
        resource_type="model",
        database="vehicle_platform",
        schema="public",
        alias="rpt_vehicle_summary",
        materialization="table",
    )

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

    with patch(
        "common.builders.reader_builder.Settings.snowflake.options",
        return_value={
            "sfURL": "test.snowflakecomputing.com",
            "sfUser": "test_user",
        },
    ):
        reader = ReaderBuilder.build(
            {
                "reader": {
                    "type": "snowflake",
                    "table": "rpt_vehicle_summary",
                }
            }
        )

    assert isinstance(reader, SnowflakeReader)
    assert reader.table == "rpt_vehicle_summary"
    assert reader.options["sfURL"] == "test.snowflakecomputing.com"

    spark = Mock()

    spark.read.format.return_value.options.return_value.option.return_value.load.return_value = (
        dataframe
    )

    documents = DbtModelDocumentLoader(
        model=model,
        reader=reader,
        id_fn=lambda row: (f"{model.name}:" f"{row['battery_health']}:" f"{row['fuel_health']}"),
        content_fn=lambda row: (
            f"Snowflake fleet summary: "
            f"{row['vehicle_count']} vehicles with "
            f"average speed {row['avg_speed']} and "
            f"battery health {row['battery_health']} "
            f"and fuel health {row['fuel_health']}."
        ),
        metadata_fn=lambda row: {
            "source": "snowflake",
            "dataset": model.name,
            "battery_health": row["battery_health"],
            "fuel_health": row["fuel_health"],
        },
    )

    documents = documents.load(spark)

    assert len(documents) == 1
    assert documents[0].id == "rpt_vehicle_summary:GOOD:GOOD"
    assert documents[0].metadata["source"] == "snowflake"
    assert documents[0].metadata["dbt_model"] == "rpt_vehicle_summary"
    assert documents[0].metadata["dbt_database"] == "vehicle_platform"

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
    assert results[0].chunk.metadata["source"] == "snowflake"
