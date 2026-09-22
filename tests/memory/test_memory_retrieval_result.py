from datetime import UTC, datetime

import pytest

from memory.models import MemoryItem
from memory.retrieval.contracts import MemoryRetrievalResult


def _memory_item() -> MemoryItem:
    return MemoryItem(
        id="memory-1",
        memory_type="semantic",
        content="Vehicle warranty information.",
        namespace="test",
        created_at=datetime.now(UTC),
    )


def test_memory_retrieval_result_wraps_memory_without_mutating_memory_item():
    item = _memory_item()

    result = MemoryRetrievalResult(
        item=item,
        retrieval_method="semantic",
        rank=1,
        retrieval_score=0.91,
        provenance={"source": "postgresql"},
    )

    assert result.item is item
    assert result.retrieval_method == "semantic"
    assert result.rank == 1
    assert result.retrieval_score == 0.91
    assert result.reranker_score is None
    assert result.provenance == {"source": "postgresql"}

    assert not hasattr(item, "retrieval_method")
    assert not hasattr(item, "retrieval_score")
    assert not hasattr(item, "reranker_score")


def test_memory_retrieval_result_supports_reranker_score():
    result = MemoryRetrievalResult(
        item=_memory_item(),
        retrieval_method="hybrid",
        rank=2,
        retrieval_score=0.72,
        reranker_score=0.88,
    )

    assert result.retrieval_score == 0.72
    assert result.reranker_score == 0.88


def test_memory_retrieval_result_rejects_empty_method():
    with pytest.raises(ValueError, match="retrieval method"):
        MemoryRetrievalResult(
            item=_memory_item(),
            retrieval_method=" ",
            rank=1,
        )


def test_memory_retrieval_result_rejects_non_positive_rank():
    with pytest.raises(ValueError, match="retrieval rank"):
        MemoryRetrievalResult(
            item=_memory_item(),
            retrieval_method="semantic",
            rank=0,
        )


def test_memory_context_defaults_to_empty_retrieval_results():
    from memory.context.builder import MemoryContext

    context = MemoryContext(
        working=(),
        semantic=(),
        episodic=(),
    )

    assert context.working_results == ()
    assert context.semantic_results == ()
    assert context.episodic_results == ()
    assert context.all_items == ()
    assert context.is_empty is True


def test_memory_context_can_carry_retrieval_provenance_separately():
    from memory.context.builder import MemoryContext

    item = _memory_item()
    result = MemoryRetrievalResult(
        item=item,
        retrieval_method="semantic",
        rank=1,
        retrieval_score=0.95,
    )

    context = MemoryContext(
        working=(),
        semantic=(item,),
        episodic=(),
        semantic_results=(result,),
    )

    assert context.semantic == (item,)
    assert context.semantic_results == (result,)
    assert context.semantic_results[0].item is item
    assert context.all_items == (item,)
