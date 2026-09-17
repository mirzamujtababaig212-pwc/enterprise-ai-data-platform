from __future__ import annotations

import pytest

from memory.evaluation.datasets.memory_retrieval_quality import (
    MEMORY_RETRIEVAL_QUALITY_CASES,
    MEMORY_RETRIEVAL_QUALITY_ITEMS,
)
from memory.evaluation.evaluator import MemoryRetrievalEvaluator
from memory.retrieval.lexical import LexicalMemoryRetriever


class BenchmarkMemoryStore:
    def __init__(self, items):
        self._items = tuple(items)

    async def search(
        self,
        namespace,
        *,
        memory_type=None,
        limit=10,
    ):
        results = [
            item
            for item in self._items
            if item.namespace == namespace
            and (memory_type is None or item.memory_type == memory_type)
            and (item.expires_at is None)
        ]
        return tuple(
            sorted(
                results,
                key=lambda item: (-item.created_at.timestamp(), item.id),
            )[:limit]
        )


@pytest.mark.asyncio
async def test_memory_retrieval_quality_baseline():
    retriever = LexicalMemoryRetriever(BenchmarkMemoryStore(MEMORY_RETRIEVAL_QUALITY_ITEMS))
    evaluator = MemoryRetrievalEvaluator(retriever, k=5)

    result = await evaluator.evaluate(MEMORY_RETRIEVAL_QUALITY_CASES)

    assert result.evaluated_queries == len(MEMORY_RETRIEVAL_QUALITY_CASES)
    assert result.successful_queries == len(MEMORY_RETRIEVAL_QUALITY_CASES)
    assert result.failed_queries == 0

    assert result.recall_at_k >= 0.90
    assert result.precision_at_k >= 0.60
    assert result.mrr >= 0.79
    assert result.ndcg_at_k >= 0.81
    assert result.mean_latency_ms >= 0.0

    assert len(result.query_results) == len(MEMORY_RETRIEVAL_QUALITY_CASES)
