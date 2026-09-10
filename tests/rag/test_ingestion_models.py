from rag.ingestion import RAGIngestionResult


def test_rag_ingestion_result_stores_metrics():
    result = RAGIngestionResult(
        documents_processed=10,
        documents_indexed=8,
        chunks_created=24,
    )

    assert result.documents_processed == 10
    assert result.documents_indexed == 8
    assert result.chunks_created == 24


def test_rag_ingestion_result_is_immutable():
    result = RAGIngestionResult(
        documents_processed=1,
        documents_indexed=1,
        chunks_created=2,
    )

    assert result.documents_processed == 1
