from unittest.mock import Mock

import pytest

from common.provenance import snowflake_source_ref
from common.snowflake.metadata import SnowflakeTableMetadata
from rag.chunking.recursive import RecursiveChunker
from rag.embeddings.gateway import GatewayEmbeddingService
from rag.indexing import RAGIndexer
from rag.ingestion import RAGIngestionService
from rag.loaders.snowflake import SnowflakeDocumentLoader
from rag.retrieval.retriever import SemanticRetriever
from rag.stores.in_memory import InMemoryVectorStore


@pytest.mark.asyncio
async def test_snowflake_rag_vertical_slice_preserves_source_provenance():
    table_metadata = SnowflakeTableMetadata(
        full_name="VEHICLE_PLATFORM.PUBLIC.RPT_VEHICLE_SUMMARY",
        database="VEHICLE_PLATFORM",
        schema="PUBLIC",
        name="RPT_VEHICLE_SUMMARY",
        table_type="TABLE",
    )

    dataframe = Mock()
    dataframe.toLocalIterator.return_value = iter(
        [
            Mock(
                asDict=Mock(
                    return_value={
                        "vehicle_id": "V001",
                        "vehicle_count": 25,
                        "avg_speed": 48.17,
                    }
                )
            ),
            Mock(
                asDict=Mock(
                    return_value={
                        "vehicle_id": "V002",
                        "vehicle_count": 18,
                        "avg_speed": 52.41,
                    }
                )
            ),
        ]
    )

    reader = Mock()
    reader.read.return_value = dataframe

    loader = SnowflakeDocumentLoader(
        metadata=table_metadata,
        reader=reader,
        id_fn=lambda row: f"vehicle:{row['vehicle_id']}",
        content_fn=lambda row: (
            f"Vehicle {row['vehicle_id']} fleet summary: "
            f"{row['vehicle_count']} vehicles with average speed "
            f"{row['avg_speed']}."
        ),
        metadata_fn=lambda row: {
            "source": "snowflake",
            "data_layer": "warehouse",
            "dataset": table_metadata.name,
            "entity_type": "vehicle",
            "vehicle_id": row["vehicle_id"],
            "vehicle_count": row["vehicle_count"],
            "avg_speed": row["avg_speed"],
        },
    )

    spark = Mock()
    documents = loader.load(spark)

    expected_source_ref = snowflake_source_ref(table_metadata).to_dict()

    assert len(documents) == 2
    assert documents[0].id == "vehicle:V001"
    assert documents[1].id == "vehicle:V002"

    assert documents[0].metadata["source_ref"] == expected_source_ref
    assert documents[1].metadata["source_ref"] == expected_source_ref

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
        "What is the average speed for vehicle V001?",
        top_k=5,
    )

    assert results
    assert results[0].chunk.metadata["source_ref"] == expected_source_ref
    assert results[0].chunk.metadata["source"] == "snowflake"
    assert results[0].chunk.metadata["dataset"] == "RPT_VEHICLE_SUMMARY"
