from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from rag.contracts import Retriever
from rag.generation.gateway import GatewayChatService
from rag.models import RetrievalResult


@dataclass(frozen=True)
class RAGSource:
    """
    Citation information exposed to callers of the RAG system.
    """

    chunk_id: str
    document_id: str
    score: float
    content: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RAGQueryResult:
    """
    Final result returned by the RAG query pipeline.
    """

    answer: str
    sources: tuple[RAGSource, ...]
    retrieved_count: int


class RAGQueryService:
    """
    End-to-end RAG query pipeline.

    Pipeline:

        Query
          ↓
        Retriever
          ↓
        Context Assembly
          ↓
        Enterprise LLM Gateway
          ↓
        Answer + Sources
    """

    def __init__(
        self,
        retriever: Retriever,
        chat_service: GatewayChatService,
    ) -> None:
        self.retriever = retriever
        self.chat_service = chat_service

    async def query(
        self,
        query: str,
        *,
        top_k: int = 5,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        user_id: str | None = None,
    ) -> RAGQueryResult:
        if not query.strip():
            raise ValueError("Query must not be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        results = await self.retriever.retrieve(
            query,
            top_k=top_k,
        )

        prompt = self._build_prompt(
            query,
            results,
        )

        response = await self.chat_service.generate(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            user_id=user_id,
        )

        answer = str(response.get("reply", ""))

        sources = tuple(self._to_source(result) for result in results)

        return RAGQueryResult(
            answer=answer,
            sources=sources,
            retrieved_count=len(results),
        )

    @staticmethod
    def _build_prompt(
        query: str,
        results: Sequence[RetrievalResult],
    ) -> str:
        context_blocks: list[str] = []

        for index, result in enumerate(results, start=1):
            context_blocks.append(
                "\n".join(
                    [
                        f"[Source {index}]",
                        f"Document ID: {result.chunk.document_id}",
                        f"Chunk ID: {result.chunk.id}",
                        f"Relevance Score: {result.score:.6f}",
                        f"Content:\n{result.chunk.content}",
                    ]
                )
            )

        context = "\n\n".join(context_blocks)

        if not context:
            context = "[No relevant sources were retrieved.]"

        return (
            "You are an enterprise knowledge assistant.\n\n"
            "Answer the user's question using the retrieved context "
            "when it is available.\n"
            "Do not invent facts that are not supported by the context.\n"
            "If the retrieved context does not contain enough information "
            "to answer the question, say so clearly.\n\n"
            f"Retrieved Context:\n{context}\n\n"
            f"User Question:\n{query}\n\n"
            "Answer:"
        )

    @staticmethod
    def _to_source(
        result: RetrievalResult,
    ) -> RAGSource:
        return RAGSource(
            chunk_id=result.chunk.id,
            document_id=result.chunk.document_id,
            score=result.score,
            content=result.chunk.content,
            metadata=dict(result.chunk.metadata),
        )
