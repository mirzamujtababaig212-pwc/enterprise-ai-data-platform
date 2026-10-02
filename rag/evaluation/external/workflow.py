from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Protocol

from rag.contracts import Retriever

from rag.evaluation.external.models import (
    ExternalEvaluationResult,
    ExternalEvaluationSample,
)
from rag.evaluation.models import RetrievalEvaluationCase
from rag.generation.gateway import GatewayChatService
from rag.query import RAGQueryService


class ExternalEvaluationEvaluator(Protocol):
    """Evaluates provider-neutral RAG samples."""

    async def evaluate(
        self,
        samples: Sequence[ExternalEvaluationSample],
    ) -> ExternalEvaluationResult: ...


class RAGGenerationEvaluationWorkflow:
    """
    Evaluate generated RAG answers using an external evaluation provider.

    The workflow reuses the evaluation retriever and the existing RAG query
    pipeline. Raw query, context, and answer data remain ephemeral and are
    passed only to the external evaluator.
    """

    def __init__(
        self,
        *,
        chat_service: GatewayChatService,
        evaluator: ExternalEvaluationEvaluator,
        top_k: int = 5,
        min_score: float | None = None,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        self._chat_service = chat_service
        self._evaluator = evaluator
        self._top_k = top_k
        self._min_score = min_score

    async def evaluate(
        self,
        cases: Sequence[RetrievalEvaluationCase],
        *,
        retriever: Retriever,
    ) -> ExternalEvaluationResult:
        if not cases:
            raise ValueError("cases must not be empty")

        rag_query_service = RAGQueryService(
            retriever=retriever,
            chat_service=self._chat_service,
        )

        samples: list[ExternalEvaluationSample] = []

        for case in cases:
            result = await rag_query_service.query(
                case.query,
                top_k=self._top_k,
                min_score=self._min_score,
            )

            samples.append(
                ExternalEvaluationSample(
                    query=case.query,
                    retrieved_contexts=tuple(source.content for source in result.sources),
                    response=result.answer,
                )
            )

        evaluation = await self._evaluator.evaluate(samples)

        from rag.retrieval.factory import RAGRetrieverFactory

        retrieval_artifact = RAGRetrieverFactory.build_retrieval_artifact(
            retriever,
        )

        return replace(
            evaluation,
            retrieval_artifact=retrieval_artifact,
        )
