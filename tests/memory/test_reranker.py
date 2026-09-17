from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from memory.models import MemoryItem
from memory.retrieval.reranker import (
    CrossEncoderMemoryReranker,
    TokenOverlapMemoryReranker,
)


def _item(
    memory_id: str,
    content: str,
    *,
    memory_type: str = "semantic",
    namespace: str = "project-a",
) -> MemoryItem:
    return MemoryItem(
        id=memory_id,
        memory_type=memory_type,  # type: ignore[arg-type]
        content=content,
        namespace=namespace,
        created_at=datetime.now(timezone.utc),
    )


@pytest.mark.asyncio
async def test_token_overlap_memory_reranker_reorders_candidates():
    reranker = TokenOverlapMemoryReranker()

    results = await reranker.rerank(
        "electric battery",
        [
            _item("weak", "vehicle maintenance"),
            _item("battery", "battery storage system"),
            _item("electric", "electric motor"),
        ],
        top_k=3,
    )

    assert [item.id for item in results] == [
        "battery",
        "electric",
        "weak",
    ]


@pytest.mark.asyncio
async def test_token_overlap_memory_reranker_returns_requested_top_k():
    reranker = TokenOverlapMemoryReranker()

    results = await reranker.rerank(
        "vehicle",
        [
            _item("a", "vehicle battery"),
            _item("b", "vehicle motor"),
            _item("c", "vehicle brakes"),
        ],
        top_k=2,
    )

    assert len(results) == 2


@pytest.mark.asyncio
async def test_token_overlap_memory_reranker_preserves_original_order_for_equal_scores():
    reranker = TokenOverlapMemoryReranker()

    results = await reranker.rerank(
        "vehicle",
        [
            _item("first", "vehicle"),
            _item("second", "vehicle"),
        ],
        top_k=2,
    )

    assert [item.id for item in results] == [
        "first",
        "second",
    ]


@pytest.mark.asyncio
async def test_token_overlap_memory_reranker_is_case_insensitive():
    reranker = TokenOverlapMemoryReranker()

    results = await reranker.rerank(
        "Production RELEASE",
        [
            _item("match", "production release requires approval"),
            _item("partial", "production deployment"),
        ],
        top_k=2,
    )

    assert [item.id for item in results] == [
        "match",
        "partial",
    ]


@pytest.mark.asyncio
async def test_token_overlap_memory_reranker_handles_empty_candidates():
    results = await TokenOverlapMemoryReranker().rerank(
        "deployment",
        [],
        top_k=5,
    )

    assert results == ()


@pytest.mark.asyncio
async def test_token_overlap_memory_reranker_rejects_invalid_arguments():
    reranker = TokenOverlapMemoryReranker()
    candidate = _item("memory", "vehicle")

    with pytest.raises(ValueError, match="empty"):
        await reranker.rerank("", [candidate], top_k=1)

    with pytest.raises(ValueError, match="top_k"):
        await reranker.rerank("vehicle", [candidate], top_k=0)


class FakeTokenizer:
    def __init__(self, encoded):
        self.encoded = encoded

    def __call__(
        self,
        queries,
        documents,
        *,
        padding,
        truncation,
        max_length,
        return_tensors,
    ):
        self.queries = queries
        self.documents = documents
        self.padding = padding
        self.truncation = truncation
        self.max_length = max_length
        self.return_tensors = return_tensors
        return self.encoded


class FakeSession:
    def __init__(self, scores):
        self.scores = np.asarray(scores, dtype=np.float32)
        self.requested_outputs = None
        self.inputs = None

    def run(self, output_names, inputs):
        self.requested_outputs = output_names
        self.inputs = inputs
        return [self.scores]


def _make_cross_encoder_reranker(
    scores,
) -> tuple[CrossEncoderMemoryReranker, FakeTokenizer, FakeSession]:
    reranker = CrossEncoderMemoryReranker.__new__(CrossEncoderMemoryReranker)

    tokenizer = FakeTokenizer(
        encoded={
            "input_ids": np.ones((len(scores), 4), dtype=np.int64),
            "attention_mask": np.ones((len(scores), 4), dtype=np.int64),
        }
    )
    session = FakeSession(scores)

    reranker.tokenizer = tokenizer
    reranker.session = session
    reranker.max_length = 8192
    reranker._input_names = ("input_ids", "attention_mask")
    reranker._output_name = "logits"

    return reranker, tokenizer, session


@pytest.mark.asyncio
async def test_cross_encoder_memory_reranker_scores_and_orders_candidates():
    reranker, tokenizer, session = _make_cross_encoder_reranker([0.2, 0.9, 0.5])

    candidates = [
        _item("memory-a", "first memory"),
        _item("memory-b", "second memory"),
        _item("memory-c", "third memory"),
    ]

    results = await reranker.rerank(
        "test query",
        candidates,
        top_k=3,
    )

    assert [item.id for item in results] == [
        "memory-b",
        "memory-c",
        "memory-a",
    ]

    assert tokenizer.queries == [
        "test query",
        "test query",
        "test query",
    ]
    assert tokenizer.documents == [
        "first memory",
        "second memory",
        "third memory",
    ]
    assert tokenizer.padding is True
    assert tokenizer.truncation is True
    assert tokenizer.max_length == 8192
    assert tokenizer.return_tensors == "np"

    assert session.requested_outputs == ["logits"]
    assert tuple(session.inputs) == (
        "input_ids",
        "attention_mask",
    )


@pytest.mark.asyncio
async def test_cross_encoder_memory_reranker_applies_top_k():
    reranker, _, _ = _make_cross_encoder_reranker([0.2, 0.9, 0.5])

    candidates = [
        _item("memory-a", "first"),
        _item("memory-b", "second"),
        _item("memory-c", "third"),
    ]

    results = await reranker.rerank(
        "test query",
        candidates,
        top_k=2,
    )

    assert [item.id for item in results] == [
        "memory-b",
        "memory-c",
    ]


@pytest.mark.asyncio
async def test_cross_encoder_memory_reranker_preserves_original_order_for_equal_scores():
    reranker, _, _ = _make_cross_encoder_reranker([0.5, 0.5, 0.5])

    candidates = [
        _item("memory-c", "third"),
        _item("memory-a", "first"),
        _item("memory-b", "second"),
    ]

    results = await reranker.rerank(
        "test query",
        candidates,
        top_k=3,
    )

    assert [item.id for item in results] == [
        "memory-c",
        "memory-a",
        "memory-b",
    ]


@pytest.mark.asyncio
async def test_cross_encoder_memory_reranker_preserves_memory_objects():
    reranker, _, _ = _make_cross_encoder_reranker([0.9, 0.1])

    candidates = [
        _item("memory-a", "first"),
        _item("memory-b", "second"),
    ]

    results = await reranker.rerank(
        "test query",
        candidates,
        top_k=2,
    )

    assert results[0] is candidates[0]
    assert results[1] is candidates[1]


@pytest.mark.asyncio
async def test_cross_encoder_memory_reranker_handles_empty_candidates():
    reranker, _, _ = _make_cross_encoder_reranker([])

    results = await reranker.rerank(
        "test query",
        [],
        top_k=5,
    )

    assert results == ()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "top_k", "message"),
    [
        ("", 5, "Query must not be empty."),
        ("   ", 5, "Query must not be empty."),
        ("test query", 0, "top_k must be greater than zero."),
        ("test query", -1, "top_k must be greater than zero."),
    ],
)
async def test_cross_encoder_memory_reranker_rejects_invalid_arguments(
    query,
    top_k,
    message,
):
    reranker, _, _ = _make_cross_encoder_reranker([0.5])

    with pytest.raises(ValueError, match=message):
        await reranker.rerank(
            query,
            [_item("memory-a", "first")],
            top_k=top_k,
        )


@pytest.mark.asyncio
async def test_cross_encoder_memory_reranker_rejects_wrong_number_of_model_scores():
    reranker, _, _ = _make_cross_encoder_reranker([0.9])

    candidates = [
        _item("memory-a", "first"),
        _item("memory-b", "second"),
    ]

    with pytest.raises(
        ValueError,
        match="different number of scores",
    ):
        await reranker.rerank(
            "test query",
            candidates,
            top_k=2,
        )
