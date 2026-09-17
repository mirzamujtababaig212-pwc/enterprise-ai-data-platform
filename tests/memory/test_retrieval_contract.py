from __future__ import annotations

from typing import get_type_hints

from memory.retrieval.contracts import MemoryRetriever
from memory.retrieval.lexical import LexicalMemoryRetriever
from memory.retrieval.postgres_lexical import PostgreSQLLexicalMemoryRetriever
from memory.retrieval.postgres_semantic import PostgreSQLSemanticMemoryRetriever


def test_memory_retriever_contract_exposes_query_aware_retrieve():
    method = MemoryRetriever.retrieve
    hints = get_type_hints(method)

    assert "query" in hints
    assert "namespace" in hints
    assert "memory_type" in hints
    assert "top_k" in hints
    assert hints["query"] is str
    assert hints["namespace"] is str
    assert hints["top_k"] is int


def test_memory_retrievers_match_contract_signature():
    implementations = (
        LexicalMemoryRetriever.retrieve,
        PostgreSQLLexicalMemoryRetriever.retrieve,
        PostgreSQLSemanticMemoryRetriever.retrieve,
    )

    contract_parameters = get_type_hints(MemoryRetriever.retrieve)

    for implementation in implementations:
        implementation_parameters = get_type_hints(implementation)

        assert implementation_parameters["query"] is contract_parameters["query"]
        assert implementation_parameters["namespace"] is contract_parameters["namespace"]
        assert implementation_parameters["memory_type"] is contract_parameters["memory_type"]
        assert implementation_parameters["top_k"] is contract_parameters["top_k"]
