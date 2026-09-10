from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from common.builders.reader_builder import ReaderBuilder
from common.readers.snowflake_reader import SnowflakeReader
from common.snowflake.control_plane import SnowflakeControlPlaneClient
from rag.chunking.recursive import RecursiveChunker
from rag.dbt import DbtModel, DbtModelDocumentLoader
from rag.embeddings.gateway import GatewayEmbeddingService
from rag.indexing import RAGIndexer
from rag.ingestion import RAGIngestionService
from rag.retrieval.retriever import SemanticRetriever
from rag.stores.in_memory import InMemoryVectorStore


@pytest.mark.asyncio
async def test_snowflake_control_plane_metadata_composes_with_data_plane_rag():
    connection = Mock()
    cursor = Mock()

    number_type = SimpleNamespace(name="NUMBER")
    float_type = SimpleNamespace(name="FLOAT")

    cursor.describe.return_value = [
        SimpleNamespace(
            name="vehicle_count",
            type_code=number_type,
            precision=38,
            scale=0,
            is_nullable=False,
        ),
        SimpleNamespace(
            name="avg_speed",
            type_code=float_type,
            precision=None,
            scale=None,
            is_nullable=True,
        ),
    ]

    connection.cursor.return_value = cursor

    control_plane = SnowflakeControlPlaneClient(
        connection=connection,
    )

    table_metadata = control_plane.get_table_metadata("VEHICLE_PLATFORM.PUBLIC.RPT_VEHICLE_SUMMARY")

    assert table_metadata.full_name == ("VEHICLE_PLATFORM.PUBLIC.RPT_VEHICLE_SUMMARY")
    assert table_metadata.database == "VEHICLE_PLATFORM"
    assert table_metadata.schema == "PUBLIC"
    assert table_metadata.name == "RPT_VEHICLE_SUMMARY"
    assert len(table_metadata.columns) == 2

    assert table_metadata.columns[0].name == "vehicle_count"
    assert table_metadata.columns[0].type_name == "NUMBER"
    assert table_metadata.columns[0].type_text == "NUMBER(38,0)"
    assert table_metadata.columns[0].nullable is False

    assert table_metadata.columns[1].name == "avg_speed"
    assert table_metadata.columns[1].type_name == "FLOAT"
    assert table_metadata.columns[1].type_text == "FLOAT"
    assert table_metadata.columns[1].nullable is True

    cursor.describe.assert_called_once_with(
        "SELECT * FROM VEHICLE_PLATFORM.PUBLIC.RPT_VEHICLE_SUMMARY"
    )

    model = DbtModel(
        unique_id="model.vehicle_dbt.rpt_vehicle_summary",
        name="rpt_vehicle_summary",
        resource_type="model",
        database="VEHICLE_PLATFORM",
        schema="PUBLIC",
        alias="RPT_VEHICLE_SUMMARY",
        materialization="table",
    )

    dataframe = Mock()
    dataframe.toLocalIterator.return_value = iter(
        [
            Mock(
                asDict=Mock(
                    return_value={
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
                    "table": "VEHICLE_PLATFORM.PUBLIC.RPT_VEHICLE_SUMMARY",
                }
            }
        )

    assert isinstance(reader, SnowflakeReader)
    assert reader.table == "VEHICLE_PLATFORM.PUBLIC.RPT_VEHICLE_SUMMARY"
    assert reader.options["sfURL"] == "test.snowflakecomputing.com"

    spark = Mock()

    spark.read.format.return_value.options.return_value.option.return_value.load.return_value = (
        dataframe
    )

    documents = DbtModelDocumentLoader(
        model=model,
        reader=reader,
        id_fn=lambda row: f"{model.name}:{row['vehicle_count']}",
        content_fn=lambda row: (
            f"Snowflake fleet summary: "
            f"{row['vehicle_count']} vehicles with "
            f"average speed {row['avg_speed']}."
        ),
        metadata_fn=lambda row: {
            "source": "snowflake",
            "dataset": table_metadata.name,
            "database": table_metadata.database,
            "schema": table_metadata.schema,
            "table": table_metadata.name,
            "table_type": table_metadata.table_type,
            "vehicle_count": row["vehicle_count"],
            "avg_speed": row["avg_speed"],
        },
    ).load(spark)

    assert len(documents) == 1
    assert documents[0].id == "rpt_vehicle_summary:25"

    assert documents[0].metadata["source"] == "snowflake"
    assert documents[0].metadata["dataset"] == "RPT_VEHICLE_SUMMARY"
    assert documents[0].metadata["database"] == "VEHICLE_PLATFORM"
    assert documents[0].metadata["schema"] == "PUBLIC"
    assert documents[0].metadata["table"] == "RPT_VEHICLE_SUMMARY"

    assert documents[0].metadata["vehicle_count"] == 25
    assert documents[0].metadata["avg_speed"] == 48.17

    assert documents[0].metadata["dbt_model"] == "rpt_vehicle_summary"
    assert documents[0].metadata["dbt_database"] == "VEHICLE_PLATFORM"
    assert documents[0].metadata["dbt_schema"] == "PUBLIC"
    assert documents[0].metadata["dbt_alias"] == "RPT_VEHICLE_SUMMARY"
    assert documents[0].metadata["dbt_materialization"] == "table"

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
        "What is the average speed for the Snowflake fleet?",
        top_k=5,
    )

    assert results
    assert results[0].chunk.metadata["source"] == "snowflake"
    assert results[0].chunk.metadata["database"] == "VEHICLE_PLATFORM"
    assert results[0].chunk.metadata["schema"] == "PUBLIC"
    assert results[0].chunk.metadata["table"] == "RPT_VEHICLE_SUMMARY"
    assert results[0].chunk.metadata["dbt_model"] == "rpt_vehicle_summary"
