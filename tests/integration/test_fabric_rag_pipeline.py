from unittest.mock import Mock

import pytest
from common.fabric.metadata import FabricTableMetadata
from common.readers.fabric_reader import FabricReader
from rag.chunking.recursive import RecursiveChunker
from rag.embeddings.gateway import GatewayEmbeddingService
from rag.indexing import RAGIndexer
from rag.ingestion import RAGIngestionService
from rag.loaders.fabric import FabricDocumentLoader
from rag.retrieval.retriever import SemanticRetriever
from rag.stores.in_memory import InMemoryVectorStore


@pytest.mark.asyncio
async def test_fabric_direct_rag_vertical_slice_preserves_provenance():
    metadata = FabricTableMetadata(
        workspace_name="VehicleWorkspace",
        lakehouse_name="VehicleLakehouse",
        catalog_name="VehicleLakehouse.Lakehouse",
        schema_name="dbo",
        table_id="table-123",
        table_name="vehicle_events",
        table_type="MANAGED",
        location="https://onelake.example/Tables/vehicle_events",
        format="DELTA",
    )

    reader = Mock(spec=FabricReader)

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
            ),
            Mock(
                asDict=Mock(
                    return_value={
                        "vehicle_id": "VH-002",
                        "avg_speed": 52.31,
                    }
                )
            ),
        ]
    )
    reader.read.return_value = dataframe

    spark = Mock()

    documents = FabricDocumentLoader(
        metadata=metadata,
        reader=reader,
        id_fn=lambda row: f"vehicle:{row['vehicle_id']}",
        content_fn=lambda row: (
            f"Fabric vehicle {row['vehicle_id']} has an " f"average speed of {row['avg_speed']}."
        ),
        metadata_fn=lambda row: {
            "source": "fabric",
            "dataset": metadata.table_name,
            "workspace": metadata.workspace_name,
            "lakehouse": metadata.lakehouse_name,
            "vehicle_id": row["vehicle_id"],
        },
    ).load(spark)

    assert len(documents) == 2

    expected_source_ref = {
        "platform": "fabric",
        "object_type": "table",
        "object_name": "vehicle_events",
        "object_id": "table-123",
        "namespace": "VehicleWorkspace.VehicleLakehouse.dbo",
    }

    assert all(document.metadata["source_ref"] == expected_source_ref for document in documents)

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

    assert result.documents_processed == 2
    assert result.documents_indexed == 2
    assert result.chunks_created >= 2

    retriever = SemanticRetriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    results = await retriever.retrieve(
        "What is the average speed of Fabric vehicle VH-001?",
        top_k=5,
    )

    assert results

    result_metadata = results[0].chunk.metadata

    assert result_metadata["source"] == "fabric"
    assert result_metadata["workspace"] == "VehicleWorkspace"
    assert result_metadata["lakehouse"] == "VehicleLakehouse"
    assert result_metadata["dataset"] == "vehicle_events"
    assert result_metadata["source_ref"] == expected_source_ref
