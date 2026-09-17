from __future__ import annotations

import asyncio
import re
from collections.abc import Mapping, Sequence
from typing import Protocol

from rag.contracts import Retriever
from rag.governance import GovernancePolicy
from rag.models import RetrievalResult

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> frozenset[str]:
    return frozenset(token.lower() for token in _TOKEN_PATTERN.findall(text))


class Reranker(Protocol):
    """
    Reorders an existing set of retrieved candidates for a query.

    A reranker does not retrieve additional documents. It only scores and
    orders the candidates supplied to it.
    """

    async def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        *,
        top_k: int,
    ) -> Sequence[RetrievalResult]: ...


class TokenOverlapReranker:
    """
    Deterministic experimental reranker based on query/document token overlap.

    This implementation is intentionally backend-independent and has no
    knowledge of evaluation labels, benchmark metadata, or relevance grades.

    It is an experimental ranking implementation used to validate the
    reranking abstraction before introducing a learned cross-encoder or
    another model-backed reranker.
    """

    async def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        *,
        top_k: int,
    ) -> Sequence[RetrievalResult]:
        if not query.strip():
            raise ValueError("Query must not be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        query_tokens = _tokenize(query)

        scored: list[tuple[RetrievalResult, float, int]] = []

        for original_rank, result in enumerate(candidates):
            document_tokens = _tokenize(result.chunk.content)

            if not query_tokens or not document_tokens:
                overlap_score = 0.0
            else:
                overlap_score = len(query_tokens & document_tokens) / len(query_tokens)

            scored.append((result, overlap_score, original_rank))

        ordered = sorted(
            scored,
            key=lambda item: (
                -item[1],
                item[2],
                item[0].chunk.id,
            ),
        )

        return tuple(
            RetrievalResult(
                chunk=result.chunk,
                score=score,
                embedding_identity=result.embedding_identity,
            )
            for result, score, _ in ordered[:top_k]
        )


class CrossEncoderReranker:
    """
    Model-backed reranker using a local ONNX cross-encoder.

    The reranker scores the supplied query/candidate pairs jointly and
    returns the candidates ordered by cross-encoder relevance score.

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
            raise ValueError(f"ONNX reranker model is missing required inputs: {missing}")

        output_names = [item.name for item in self.session.get_outputs()]

        if not output_names:
            raise ValueError("ONNX reranker model has no outputs.")

        self._input_names = tuple(
            name for name in ("input_ids", "attention_mask") if name in input_names
        )
        self._output_name = output_names[0]

    async def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        *,
        top_k: int,
    ) -> Sequence[RetrievalResult]:
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

        return tuple(
            RetrievalResult(
                chunk=result.chunk,
                score=score,
                embedding_identity=result.embedding_identity,
            )
            for result, score, _ in ordered[:limit]
        )

    def _score_candidates(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
    ) -> list[tuple[RetrievalResult, float, int]]:
        encoded = self.tokenizer(
            [query] * len(candidates),
            [result.chunk.content for result in candidates],
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
            (result, float(score), original_rank)
            for original_rank, (result, score) in enumerate(zip(candidates, scores, strict=True))
        ]


class RerankingRetriever:
    """
    Compose an existing Retriever with a post-retrieval Reranker.

    The underlying retriever remains responsible for retrieval, filtering,
    governance, and candidate generation. The reranker only reorders the
    resulting candidate set.
    """

    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker,
        *,
        candidate_k: int | None = None,
    ) -> None:
        if candidate_k is not None and candidate_k <= 0:
            raise ValueError("candidate_k must be greater than zero.")

        self.retriever = retriever
        self.reranker = reranker
        self.candidate_k = candidate_k

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float | None = None,
        metadata_filter: Mapping[str, object] | None = None,
        governance_policy: GovernancePolicy | None = None,
    ) -> Sequence[RetrievalResult]:
        if not query.strip():
            raise ValueError("Query must not be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if min_score is not None and not -1.0 <= min_score <= 1.0:
            raise ValueError("min_score must be between -1.0 and 1.0.")

        candidate_k = self.candidate_k or top_k

        candidates = await self.retriever.retrieve(
            query,
            top_k=max(candidate_k, top_k),
            min_score=min_score,
            metadata_filter=metadata_filter,
            governance_policy=governance_policy,
        )

        return await self.reranker.rerank(
            query,
            candidates,
            top_k=top_k,
        )
