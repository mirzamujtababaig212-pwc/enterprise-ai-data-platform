from __future__ import annotations

import asyncio
from dataclasses import dataclass

import numpy as np
import pytest

from rag.models import (
    DocumentChunk,
    EmbeddingIdentity,
    RetrievalResult,
)
from rag.retrieval.reranker import CrossEncoderReranker


@dataclass
class FakeTokenizer:
    encoded: dict[str, np.ndarray]

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


def make_result(
    chunk_id: str,
    content: str,
    score: float = 0.0,
    embedding_identity: EmbeddingIdentity | None = None,
) -> RetrievalResult:
    return RetrievalResult(
        chunk=DocumentChunk(
            id=chunk_id,
            document_id=f"doc-{chunk_id}",
            content=content,
        ),
        score=score,
        embedding_identity=embedding_identity,
    )


def make_reranker(
    scores,
) -> tuple[CrossEncoderReranker, FakeTokenizer, FakeSession]:
    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)

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


def test_rerank_scores_and_orders_candidates():
    reranker, tokenizer, session = make_reranker([0.2, 0.9, 0.5])

    candidates = [
        make_result("chunk-a", "first"),
        make_result("chunk-b", "second"),
        make_result("chunk-c", "third"),
    ]

    result = asyncio.run(
        reranker.rerank(
            "test query",
            candidates,
            top_k=3,
        )
    )

    assert [item.chunk.id for item in result] == [
        "chunk-b",
        "chunk-c",
        "chunk-a",
    ]
    assert [item.score for item in result] == pytest.approx([0.9, 0.5, 0.2])

    assert tokenizer.queries == [
        "test query",
        "test query",
        "test query",
    ]
    assert tokenizer.documents == [
        "first",
        "second",
        "third",
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


def test_rerank_preserves_original_order_for_equal_scores():
    reranker, _, _ = make_reranker([0.5, 0.5, 0.5])

    candidates = [
        make_result("chunk-c", "third"),
        make_result("chunk-a", "first"),
        make_result("chunk-b", "second"),
    ]

    result = asyncio.run(
        reranker.rerank(
            "test query",
            candidates,
            top_k=3,
        )
    )

    assert [item.chunk.id for item in result] == [
        "chunk-c",
        "chunk-a",
        "chunk-b",
    ]


def test_rerank_applies_top_k():
    reranker, _, _ = make_reranker([0.2, 0.9, 0.5])

    candidates = [
        make_result("chunk-a", "first"),
        make_result("chunk-b", "second"),
        make_result("chunk-c", "third"),
    ]

    result = asyncio.run(
        reranker.rerank(
            "test query",
            candidates,
            top_k=2,
        )
    )

    assert [item.chunk.id for item in result] == [
        "chunk-b",
        "chunk-c",
    ]


def test_rerank_preserves_embedding_identity():
    identity = EmbeddingIdentity(
        requested_provider="test",
        requested_model="test-model",
        resolved_provider="test",
        resolved_model="test-model",
        dimension=3,
    )

    reranker, _, _ = make_reranker([0.9])

    candidate = make_result(
        "chunk-a",
        "first",
        embedding_identity=identity,
    )

    result = asyncio.run(
        reranker.rerank(
            "test query",
            [candidate],
            top_k=1,
        )
    )

    assert result[0].embedding_identity == identity


def test_rerank_empty_candidates_returns_empty():
    reranker, _, _ = make_reranker([])

    result = asyncio.run(
        reranker.rerank(
            "test query",
            [],
            top_k=5,
        )
    )

    assert result == ()


@pytest.mark.parametrize(
    ("query", "top_k", "message"),
    [
        ("", 5, "Query must not be empty."),
        ("   ", 5, "Query must not be empty."),
        ("test query", 0, "top_k must be greater than zero."),
        ("test query", -1, "top_k must be greater than zero."),
    ],
)
def test_rerank_rejects_invalid_arguments(query, top_k, message):
    reranker, _, _ = make_reranker([0.5])

    with pytest.raises(ValueError, match=message):
        asyncio.run(
            reranker.rerank(
                query,
                [make_result("chunk-a", "first")],
                top_k=top_k,
            )
        )


def test_rerank_rejects_wrong_number_of_model_scores():
    reranker, _, _ = make_reranker([0.9])

    candidates = [
        make_result("chunk-a", "first"),
        make_result("chunk-b", "second"),
    ]

    with pytest.raises(
        ValueError,
        match="different number of scores",
    ):
        asyncio.run(
            reranker.rerank(
                "test query",
                candidates,
                top_k=2,
            )
        )


def test_rerank_runs_scoring_without_changing_candidate_objects():
    reranker, _, _ = make_reranker([0.9, 0.1])

    candidates = [
        make_result("chunk-a", "first", score=0.25),
        make_result("chunk-b", "second", score=0.75),
    ]

    original_chunks = [item.chunk for item in candidates]

    result = asyncio.run(
        reranker.rerank(
            "test query",
            candidates,
            top_k=2,
        )
    )

    assert [item.chunk for item in candidates] == original_chunks
    assert [item.chunk.id for item in result] == [
        "chunk-a",
        "chunk-b",
    ]
