from unittest.mock import Mock

import pytest

from common.databricks.metadata import DatabricksTableMetadata
from common.provenance import databricks_source_ref
from common.readers.databricks_reader import DatabricksReader
from rag.chunking.recursive import RecursiveChunker
from rag.embeddings.gateway import GatewayEmbeddingService
from rag.indexing import RAGIndexer
from rag.ingestion import RAGIngestionService
from rag.loaders.databricks import DatabricksDocumentLoader
from rag.retrieval.retriever import SemanticRetriever
from rag.stores.in_memory import InMemoryVectorStore


@pytest.mark.asyncio
async def test_databricks_direct_rag_vertical_slice_preserves_provenance():
    table_metadata = DatabricksTableMetadata(
        full_name="vehicle_platform.analytics.vehicle_events",
        catalog="vehicle_platform",
        schema="analytics",
        name="vehicle_events",
        table_type="MANAGED",
        table_id="table-123",
    )

    reader = Mock(spec=DatabricksReader)

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

    loader = DatabricksDocumentLoader(
        metadata=table_metadata,
        reader=reader,
        id_fn=lambda row: f"vehicle:{row['vehicle_id']}",
        content_fn=lambda row: (
            f"Databricks vehicle {row['vehicle_id']} has an "
            f"average speed of {row['avg_speed']}."
        ),
        metadata_fn=lambda row: {
            "source": "databricks",
            "data_layer": "warehouse",
            "dataset": table_metadata.name,
            "entity_type": "vehicle",
            "vehicle_id": row["vehicle_id"],
        },
    )

    documents = loader.load(spark)

    expected_source_ref = databricks_source_ref(table_metadata).to_dict()

    assert len(documents) == 2
    assert documents[0].id == "vehicle:VH-001"
    assert documents[1].id == "vehicle:VH-002"

    assert documents[0].metadata["source"] == "databricks"
    assert documents[0].metadata["dataset"] == "vehicle_events"
    assert documents[0].metadata["entity_type"] == "vehicle"
    assert documents[0].metadata["vehicle_id"] == "VH-001"

    assert documents[0].metadata["source_ref"] == expected_source_ref
    assert documents[1].metadata["source_ref"] == expected_source_ref

    reader.read.assert_called_once_with(spark)

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
        "What is the average speed of Databricks vehicle VH-001?",
        top_k=5,
    )

    assert results

    result_metadata = results[0].chunk.metadata

    assert result_metadata["source"] == "databricks"
    assert result_metadata["dataset"] == "vehicle_events"
    assert result_metadata["entity_type"] == "vehicle"
    assert result_metadata["source_ref"] == expected_source_ref
