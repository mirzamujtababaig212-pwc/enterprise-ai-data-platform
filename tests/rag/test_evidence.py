from ai_platform.agents.evaluation.evidence import _extract_rag_source


def test_rag_evidence_source_preserves_canonical_provenance():
    payload = {
        "content": "Customer data is stored in the analytics platform.",
        "chunk_id": "customer-doc:chunk:0",
        "document_id": "customer-doc",
        "score": 0.92,
        "retrieval_score": 0.81,
        "reranker_score": 0.92,
        "retrieval_rank": 1,
        "retrieval_method": "hybrid+rereanked",
        "source_ref": {
            "platform": "snowflake",
            "object_type": "table",
            "object_name": "ANALYTICS.CUSTOMERS",
            "namespace": "ANALYTICS",
            "environment": "prod",
        },
        "locator": {
            "type": "document_section",
            "value": "customer_overview",
        },
    }

    source = _extract_rag_source(payload, source_index=0)

    assert source is not None
    assert source.evidence_id == "evidence:customer-doc:chunk:0"
    assert source.source_index == 0
    assert source.retrieval_rank == 1
    assert source.retrieval_method == "hybrid+rereanked"
    assert source.retrieval_score == 0.81
    assert source.reranker_score == 0.92
    assert source.source_ref == payload["source_ref"]
    assert source.locator == payload["locator"]


def test_rag_evidence_source_defaults_rank_from_source_index():
    payload = {
        "content": "Example content",
        "chunk_id": "doc:chunk:4",
        "document_id": "doc",
        "score": 0.75,
    }

    source = _extract_rag_source(payload, source_index=4)

    assert source is not None
    assert source.retrieval_rank == 5
    assert source.retrieval_method is None
    assert source.source_ref is None
    assert source.locator is None
    assert source.retrieval_score == 0.75
    assert source.reranker_score is None


def test_rag_evidence_source_rejects_empty_content():
    payload = {
        "content": "   ",
        "chunk_id": "doc:chunk:0",
    }

    assert _extract_rag_source(payload, source_index=0) is None
