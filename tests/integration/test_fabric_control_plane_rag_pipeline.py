from unittest.mock import Mock

import pytest

from common.fabric.control_plane import FabricControlPlaneClient
from common.readers.fabric_reader import FabricReader
from rag.chunking.recursive import RecursiveChunker
from rag.dbt import DbtModel, DbtModelDocumentLoader
from rag.embeddings.gateway import GatewayEmbeddingService
from rag.indexing import RAGIndexer
from rag.ingestion import RAGIngestionService
from rag.retrieval.retriever import SemanticRetriever
from rag.stores.in_memory import InMemoryVectorStore


@pytest.mark.asyncio
async def test_fabric_control_plane_metadata_composes_with_data_plane_rag():
    def fake_get(path, params, access_token):
        assert path == (
            "/VehicleWorkspace/VehicleLakehouse.Lakehouse"
            "/api/2.1/unity-catalog/tables/"
            "VehicleLakehouse.Lakehouse.dbo.vehicle_events"
        )
        assert params is None
        assert access_token == "test-token"

        return {
            "name": "vehicle_events",
            "catalog_name": "VehicleLakehouse.Lakehouse",
            "schema_name": "dbo",
            "table_type": "MANAGED",
            "data_source_format": "DELTA",
            "storage_location": ("https://onelake.example/Tables/vehicle_events"),
            "comment": "Vehicle telemetry",
            "owner": "data-platform",
            "table_id": "table-123",
            "properties": {
                "domain": "vehicle",
                "tier": "gold",
            },
            "columns": [
                {
                    "name": "vehicle_id",
                    "type_name": "string",
                    "type_text": "string",
                    "nullable": False,
                    "comment": "Vehicle identifier",
                    "position": 0,
                    "partition_index": None,
                },
                {
                    "name": "avg_speed",
                    "type_name": "double",
                    "type_text": "double",
                    "nullable": True,
                    "comment": "Average speed",
                    "position": 1,
                    "partition_index": None,
                },
            ],
        }

    control_plane = FabricControlPlaneClient(
        workspace="VehicleWorkspace",
        lakehouse="VehicleLakehouse",
        access_token="test-token",
        request_get=fake_get,
    )

    table_metadata = control_plane.get_table_metadata(
        table_name="vehicle_events",
        schema_name="dbo",
    )

    assert table_metadata.workspace_id is None
    assert table_metadata.workspace_name == "VehicleWorkspace"
    assert table_metadata.lakehouse_id is None
    assert table_metadata.lakehouse_name == "VehicleLakehouse"

    assert table_metadata.catalog_name == "VehicleLakehouse.Lakehouse"
    assert table_metadata.schema_name == "dbo"
    assert table_metadata.table_name == "vehicle_events"
    assert table_metadata.table_type == "MANAGED"
    assert table_metadata.table_id == "table-123"
    assert table_metadata.format == "DELTA"
    assert table_metadata.location == ("https://onelake.example/Tables/vehicle_events")
    assert table_metadata.owner == "data-platform"
    assert table_metadata.comment == "Vehicle telemetry"
    assert table_metadata.properties == {
        "domain": "vehicle",
        "tier": "gold",
    }

    assert len(table_metadata.columns) == 2
    assert table_metadata.columns[0].name == "vehicle_id"
    assert table_metadata.columns[0].type_name == "string"
    assert table_metadata.columns[0].nullable is False

    assert table_metadata.columns[1].name == "avg_speed"
    assert table_metadata.columns[1].type_name == "double"
    assert table_metadata.columns[1].nullable is True

    model = DbtModel(
        unique_id="model.vehicle_fabric.rpt_vehicle_events",
        name="rpt_vehicle_events",
        resource_type="model",
        database="VehicleLakehouse.Lakehouse",
        schema="dbo",
        alias="vehicle_events",
        materialization="table",
    )

    reader = FabricReader(
        table="VehicleLakehouse.Lakehouse.dbo.vehicle_events",
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
    spark.read.format.return_value.load.return_value = dataframe

    documents = DbtModelDocumentLoader(
        model=model,
        reader=reader,
        id_fn=lambda row: f"{model.name}:{row['vehicle_id']}",
        content_fn=lambda row: (
            f"Fabric vehicle {row['vehicle_id']} has an " f"average speed of {row['avg_speed']}."
        ),
        metadata_fn=lambda row: {
            "source": "fabric",
            "dataset": table_metadata.table_name,
            "workspace": table_metadata.workspace_name,
            "lakehouse": table_metadata.lakehouse_name,
            "catalog": table_metadata.catalog_name,
            "schema": table_metadata.schema_name,
            "table": table_metadata.table_name,
            "table_id": table_metadata.table_id,
            "table_type": table_metadata.table_type,
            "format": table_metadata.format,
            "vehicle_id": row["vehicle_id"],
        },
    ).load(spark)

    spark.read.format.assert_called_once_with("delta")

    dataframe.toLocalIterator.assert_called_once()

    assert len(documents) == 1
    assert documents[0].id == "rpt_vehicle_events:VH-001"

    assert documents[0].metadata["source"] == "fabric"
    assert documents[0].metadata["dataset"] == "vehicle_events"
    assert documents[0].metadata["workspace"] == "VehicleWorkspace"
    assert documents[0].metadata["lakehouse"] == "VehicleLakehouse"
    assert documents[0].metadata["catalog"] == "VehicleLakehouse.Lakehouse"
    assert documents[0].metadata["schema"] == "dbo"
    assert documents[0].metadata["table"] == "vehicle_events"
    assert documents[0].metadata["table_id"] == "table-123"
    assert documents[0].metadata["table_type"] == "MANAGED"
    assert documents[0].metadata["format"] == "DELTA"
    assert documents[0].metadata["vehicle_id"] == "VH-001"

    assert documents[0].metadata["dbt_unique_id"] == ("model.vehicle_fabric.rpt_vehicle_events")
    assert documents[0].metadata["dbt_model"] == "rpt_vehicle_events"
    assert documents[0].metadata["dbt_database"] == "VehicleLakehouse.Lakehouse"
    assert documents[0].metadata["dbt_schema"] == "dbo"
    assert documents[0].metadata["dbt_alias"] == "vehicle_events"
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
        "What is the average speed of Fabric vehicle VH-001?",
        top_k=5,
    )

    assert results
    assert results[0].chunk.metadata["source"] == "fabric"
    assert results[0].chunk.metadata["workspace"] == "VehicleWorkspace"
    assert results[0].chunk.metadata["lakehouse"] == "VehicleLakehouse"
    assert results[0].chunk.metadata["catalog"] == "VehicleLakehouse.Lakehouse"
    assert results[0].chunk.metadata["schema"] == "dbo"
    assert results[0].chunk.metadata["table"] == "vehicle_events"
    assert results[0].chunk.metadata["table_id"] == "table-123"
    assert results[0].chunk.metadata["dbt_model"] == "rpt_vehicle_events"
