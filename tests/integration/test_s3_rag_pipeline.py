from __future__ import annotations

from unittest.mock import Mock

import pytest

from ai_platform.llm_gateway.routing.router import Router
from rag.chunking.recursive import RecursiveChunker
from rag.embeddings.gateway import GatewayEmbeddingService
from rag.indexing import RAGIndexer
from rag.loaders import S3DocumentLoader
from rag.retrieval import SemanticRetriever
from rag.stores import InMemoryVectorStore


@pytest.mark.asyncio
async def test_s3_dataset_flows_into_rag_retrieval():
    dataframe = Mock()
    dataframe.toLocalIterator.return_value = iter(
        [
            Mock(
                asDict=Mock(
                    return_value={
                        "vehicle_id": "V001",
                        "event_count": 3,
                        "avg_speed": 48.17,
                    }
                )
            ),
            Mock(
                asDict=Mock(
                    return_value={
                        "vehicle_id": "V002",
                        "event_count": 2,
                        "avg_speed": 34.73,
                    }
                )
            ),
        ]
    )

    reader = Mock()
    reader.read.return_value = dataframe

    loader = S3DocumentLoader(
        reader=reader,
        path="s3://test-bucket/bronze/vehicle_events",
        id_fn=lambda row: f"vehicle:{row['vehicle_id']}",
        content_fn=lambda row: (
            f"Vehicle {row['vehicle_id']} recorded "
            f"{row['event_count']} events. "
            f"Average speed was {row['avg_speed']:.2f}."
        ),
        metadata_fn=lambda row: {
            "source": "s3.vehicle_events",
            "data_layer": "bronze",
            "dataset": "vehicle_events",
            "entity_type": "vehicle",
            "vehicle_id": row["vehicle_id"],
        },
    )

    documents = loader.load(Mock())

    assert {document.id for document in documents} == {
        "vehicle:V001",
        "vehicle:V002",
    }

    expected_source_ref = {
        "platform": "aws",
        "object_type": "s3_path",
        "object_name": "s3://test-bucket/bronze/vehicle_events",
        "namespace": "s3://test-bucket",
    }

    documents_by_id = {document.id: document for document in documents}

    assert documents_by_id["vehicle:V001"].metadata["source_ref"] == expected_source_ref
    assert documents_by_id["vehicle:V001"].metadata["vehicle_id"] == "V001"
    assert documents_by_id["vehicle:V001"].content == (
        "Vehicle V001 recorded 3 events. Average speed was 48.17."
    )

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
        top_k=2,
    )

    assert results
    assert results[0].chunk.document_id == "vehicle:V001"
    assert "Average speed was 48.17." in results[0].chunk.content
    assert results[0].chunk.metadata["source"] == "s3.vehicle_events"
    assert results[0].chunk.metadata["data_layer"] == "bronze"
    assert results[0].chunk.metadata["dataset"] == "vehicle_events"
    assert results[0].chunk.metadata["vehicle_id"] == "V001"
    assert results[0].chunk.metadata["source_ref"] == expected_source_ref
