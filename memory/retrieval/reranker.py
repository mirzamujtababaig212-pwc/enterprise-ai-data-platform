from __future__ import annotations

import asyncio
import re
from collections.abc import Sequence
from typing import Protocol

from memory.models import MemoryItem

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> frozenset[str]:
    return frozenset(token.lower() for token in _TOKEN_PATTERN.findall(text))


class MemoryReranker(Protocol):
    """
    Reorders an existing set of memory candidates for a query.

    A memory reranker does not retrieve additional memories. It only
    scores and orders the candidates supplied to it.
    """

    async def rerank(
        self,
        query: str,
        candidates: Sequence[MemoryItem],
        *,
        top_k: int,
    ) -> Sequence[MemoryItem]: ...


class TokenOverlapMemoryReranker:
    """
    Deterministic reranker based on query/memory token overlap.

    This implementation is backend-independent and has no knowledge of
    retrieval backends or evaluation labels.
    """

    async def rerank(
        self,
        query: str,
        candidates: Sequence[MemoryItem],
        *,
        top_k: int,
    ) -> Sequence[MemoryItem]:
        if not query.strip():
            raise ValueError("Query must not be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        query_tokens = _tokenize(query)

        scored: list[tuple[MemoryItem, float, int]] = []

        for original_rank, item in enumerate(candidates):
            memory_tokens = _tokenize(item.content)

            if not query_tokens or not memory_tokens:
                overlap_score = 0.0
            else:
                overlap_score = len(query_tokens & memory_tokens) / len(query_tokens)

            scored.append((item, overlap_score, original_rank))

        ordered = sorted(
            scored,
            key=lambda item: (
                -item[1],
                item[2],
                item[0].id,
            ),
        )

        return tuple(item[0] for item in ordered[:top_k])


class CrossEncoderMemoryReranker:
    """
    Model-backed memory reranker using a local ONNX cross-encoder.

    The reranker scores the supplied query/memory pairs jointly and returns
    the original memory items ordered by cross-encoder relevance score.

    Model loading is performed once during construction. Tokenization and
    inference are moved to a worker thread so synchronous ONNX execution
    does not block the asyncio event loop.
    """

    DEFAULT_MODEL_ID = "jinaai/jina-reranker-v1-tiny-en"
    DEFAULT_ONNX_FILENAME = "onnx/model_int8.onnx"

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        onnx_filename: str = DEFAULT_ONNX_FILENAME,
        revision: str | None = None,
        max_length: int = 8192,
    ) -> None:
        if not model_id.strip():
            raise ValueError("model_id must not be empty.")

        if not onnx_filename.strip():
            raise ValueError("onnx_filename must not be empty.")

        if max_length <= 0:
            raise ValueError("max_length must be greater than zero.")

        from huggingface_hub import hf_hub_download
        from onnxruntime import InferenceSession, SessionOptions
        from transformers import AutoTokenizer

        self.model_id = model_id
        self.onnx_filename = onnx_filename
        self.revision = revision
        self.max_length = max_length

        tokenizer_kwargs = {
            "trust_remote_code": False,
        }

        if revision is not None:
            tokenizer_kwargs["revision"] = revision

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            **tokenizer_kwargs,
        )

        model_path = hf_hub_download(
            repo_id=model_id,
            filename=onnx_filename,
            revision=revision,
        )

        session_options = SessionOptions()
        self.session = InferenceSession(
            model_path,
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )

        input_names = {item.name for item in self.session.get_inputs()}

        required_inputs = {"input_ids", "attention_mask"}
        missing_inputs = required_inputs - input_names

        if missing_inputs:
            missing = ", ".join(sorted(missing_inputs))
            raise ValueError(f"ONNX memory reranker model is missing required inputs: {missing}")

        output_names = [item.name for item in self.session.get_outputs()]

        if not output_names:
            raise ValueError("ONNX memory reranker model has no outputs.")

        self._input_names = tuple(
            name for name in ("input_ids", "attention_mask") if name in input_names
        )
        self._output_name = output_names[0]

    async def rerank(
        self,
        query: str,
        candidates: Sequence[MemoryItem],
        *,
        top_k: int,
    ) -> Sequence[MemoryItem]:
        if not query.strip():
            raise ValueError("Query must not be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if not candidates:
            return ()

        limit = min(top_k, len(candidates))

        scored = await asyncio.to_thread(
            self._score_candidates,
            query,
            candidates,
        )

        ordered = sorted(
            scored,
            key=lambda item: (
                -item[1],
                item[2],
            ),
        )

        return tuple(item[0] for item in ordered[:limit])

    def _score_candidates(
        self,
        query: str,
        candidates: Sequence[MemoryItem],
    ) -> list[tuple[MemoryItem, float, int]]:
        encoded = self.tokenizer(
            [query] * len(candidates),
            [item.content for item in candidates],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )

        inputs = {name: encoded[name] for name in self._input_names if name in encoded}

        output = self.session.run(
            [self._output_name],
            inputs,
        )[0]

        scores = output.reshape(-1)

        if len(scores) != len(candidates):
            raise ValueError(
                "ONNX reranker returned a different number of scores " "than candidates."
            )

        return [
            (item, float(score), original_rank)
            for original_rank, (item, score) in enumerate(zip(candidates, scores, strict=True))
        ]
