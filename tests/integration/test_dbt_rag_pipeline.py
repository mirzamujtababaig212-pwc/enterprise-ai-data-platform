from unittest.mock import Mock

import pytest

from rag.dbt import (
    DbtManifestParser,
    DbtModelDocumentLoader,
    DbtModelSelectionPolicy,
)
from rag.embeddings.gateway import GatewayEmbeddingService
from rag.indexing import RAGIndexer
from rag.ingestion import RAGIngestionService
from rag.retrieval.retriever import SemanticRetriever
from rag.stores.in_memory import InMemoryVectorStore
from rag.chunking.recursive import RecursiveChunker


@pytest.mark.asyncio
async def test_dbt_selected_model_flows_through_rag_pipeline():
    manifest = {
        "nodes": {
            "model.vehicle_dbt.rpt_vehicle_summary": {
                "resource_type": "model",
                "name": "rpt_vehicle_summary",
                "database": "vehicle_platform",
                "schema": "public",
                "alias": "rpt_vehicle_summary",
                "description": "Fleet vehicle summary",
                "config": {
                    "materialized": "table",
                },
                "tags": ["ai_knowledge"],
            },
            "model.vehicle_dbt.dbt_spark_metastore_test": {
                "resource_type": "model",
                "name": "dbt_spark_metastore_test",
                "config": {
                    "materialized": "table",
                },
            },
        }
    }

    models = DbtManifestParser().parse(manifest)

    policy = DbtModelSelectionPolicy()

    selected_models = [model for model in models if policy.select(model)]

    assert [model.name for model in selected_models] == ["rpt_vehicle_summary"]

    model = selected_models[0]

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

    reader = Mock()
    reader.read.return_value = dataframe

    loader = DbtModelDocumentLoader(
        model=model,
        reader=reader,
        id_fn=lambda row: (f"{model.name}:" f"{row['battery_health']}:" f"{row['fuel_health']}"),
        content_fn=lambda row: (
            f"Fleet summary for battery health "
            f"{row['battery_health']} and fuel health "
            f"{row['fuel_health']}: "
            f"{row['vehicle_count']} vehicles with "
            f"average speed {row['avg_speed']}."
        ),
        metadata_fn=lambda row: {
            "battery_health": row["battery_health"],
            "fuel_health": row["fuel_health"],
            "source": model.name,
        },
    )

    spark = Mock()

    documents = loader.load(spark)

    assert len(documents) == 1
    assert documents[0].metadata["dbt_model"] == ("rpt_vehicle_summary")

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
        "Which fleet vehicles have good battery and fuel health?",
        top_k=5,
    )

    assert results
    assert results[0].chunk.metadata["dbt_model"] == ("rpt_vehicle_summary")
