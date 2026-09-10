from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from common.databricks.control_plane import DatabricksControlPlaneClient
from common.readers.databricks_reader import DatabricksReader
from rag.chunking.recursive import RecursiveChunker
from rag.dbt import DbtModel, DbtModelDocumentLoader
from rag.embeddings.gateway import GatewayEmbeddingService
from rag.indexing import RAGIndexer
from rag.ingestion import RAGIngestionService
from rag.retrieval.retriever import SemanticRetriever
from rag.stores.in_memory import InMemoryVectorStore


@pytest.mark.asyncio
async def test_databricks_control_plane_metadata_composes_with_data_plane_rag():
    client = Mock()

    table = SimpleNamespace(
        full_name="vehicle_platform.analytics.rpt_vehicle_summary",
        catalog_name="vehicle_platform",
        schema_name="analytics",
        name="rpt_vehicle_summary",
        table_type=SimpleNamespace(value="MANAGED"),
        owner="data-platform",
        comment="Vehicle fleet summary",
        table_id="table-123",
        properties={"domain": "vehicle", "tier": "gold"},
        columns=[
            SimpleNamespace(
                name="vehicle_id",
                type_name=SimpleNamespace(value="STRING"),
                type_text="string",
                nullable=False,
                comment="Vehicle identifier",
                position=0,
                partition_index=None,
            ),
            SimpleNamespace(
                name="avg_speed",
                type_name=SimpleNamespace(value="DOUBLE"),
                type_text="double",
                nullable=True,
                comment="Average speed",
                position=1,
                partition_index=None,
            ),
        ],
    )

    client.tables.get.return_value = table

    control_plane = DatabricksControlPlaneClient(client=client)

    table_metadata = control_plane.get_table_metadata(
        "vehicle_platform.analytics.rpt_vehicle_summary"
    )

    assert table_metadata.full_name == "vehicle_platform.analytics.rpt_vehicle_summary"
    assert table_metadata.catalog == "vehicle_platform"
    assert table_metadata.schema == "analytics"
    assert table_metadata.name == "rpt_vehicle_summary"
    assert table_metadata.table_type == "MANAGED"
    assert table_metadata.table_id == "table-123"
    assert len(table_metadata.columns) == 2

    model = DbtModel(
        unique_id="model.vehicle_dbt.rpt_vehicle_summary",
        name="rpt_vehicle_summary",
        resource_type="model",
        database="vehicle_platform",
        schema="analytics",
        alias="rpt_vehicle_summary",
        materialization="table",
    )

    reader = DatabricksReader(
        table="vehicle_platform.analytics.rpt_vehicle_summary",
    )

    dataframe = Mock()
    dataframe.toLocalIterator.return_value = iter(
        [
            Mock(
                asDict=Mock(
                    return_value={
                        "vehicle_id": "VH-001",
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
        id_fn=lambda row: f"{model.name}:{row['vehicle_id']}",
        content_fn=lambda row: (
            f"Vehicle {row['vehicle_id']} has an average speed " f"of {row['avg_speed']}."
        ),
        metadata_fn=lambda row: {
            "source": "databricks",
            "dataset": table_metadata.name,
            "catalog": table_metadata.catalog,
            "schema": table_metadata.schema,
            "table": table_metadata.name,
            "table_id": table_metadata.table_id,
            "table_type": table_metadata.table_type,
            "vehicle_id": row["vehicle_id"],
        },
    ).load(spark)

    client.tables.get.assert_called_once_with(
        "vehicle_platform.analytics.rpt_vehicle_summary",
    )

    spark.table.assert_called_once_with(
        "vehicle_platform.analytics.rpt_vehicle_summary",
    )

    assert len(documents) == 1
    assert documents[0].id == "rpt_vehicle_summary:VH-001"

    assert documents[0].metadata["source"] == "databricks"
    assert documents[0].metadata["dataset"] == "rpt_vehicle_summary"
    assert documents[0].metadata["catalog"] == "vehicle_platform"
    assert documents[0].metadata["schema"] == "analytics"
    assert documents[0].metadata["table"] == "rpt_vehicle_summary"
    assert documents[0].metadata["table_id"] == "table-123"
    assert documents[0].metadata["table_type"] == "MANAGED"
    assert documents[0].metadata["vehicle_id"] == "VH-001"

    assert documents[0].metadata["dbt_unique_id"] == ("model.vehicle_dbt.rpt_vehicle_summary")
    assert documents[0].metadata["dbt_model"] == "rpt_vehicle_summary"
    assert documents[0].metadata["dbt_database"] == "vehicle_platform"
    assert documents[0].metadata["dbt_schema"] == "analytics"
    assert documents[0].metadata["dbt_alias"] == "rpt_vehicle_summary"
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
        "What is the average speed of vehicle VH-001?",
        top_k=5,
    )

    assert results
    assert results[0].chunk.metadata["source"] == "databricks"
    assert results[0].chunk.metadata["catalog"] == "vehicle_platform"
    assert results[0].chunk.metadata["schema"] == "analytics"
    assert results[0].chunk.metadata["table"] == "rpt_vehicle_summary"
    assert results[0].chunk.metadata["table_id"] == "table-123"
    assert results[0].chunk.metadata["dbt_model"] == "rpt_vehicle_summary"
