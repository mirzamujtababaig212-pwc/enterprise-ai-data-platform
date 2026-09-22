from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from ai_platform.agents.observability import (
    AgentExecutionEvent,
    AgentExecutionEventType,
)
from ai_platform.agents.observer import AgentExecutionObserver
from memory.models import MemoryItem
from memory.retrieval.contracts import MemoryRetrievalResult, MemoryRetriever
from memory.service import MemoryService


@dataclass(frozen=True)
class MemoryContext:
    working: tuple[MemoryItem, ...]
    semantic: tuple[MemoryItem, ...]
    episodic: tuple[MemoryItem, ...]
    working_results: tuple[MemoryRetrievalResult, ...] = ()
    semantic_results: tuple[MemoryRetrievalResult, ...] = ()
    episodic_results: tuple[MemoryRetrievalResult, ...] = ()

    @property
    def all_items(self) -> tuple[MemoryItem, ...]:
        return self.working + self.semantic + self.episodic

    @property
    def is_empty(self) -> bool:
        return not self.all_items


class MemoryContextBuilder:
    """
    Builds a structured context from the agent's memories.

    The builder intentionally depends on MemoryService rather than
    directly accessing a concrete memory store.
    """

    def __init__(
        self,
        memory_service: MemoryService,
        *,
        memory_retriever: MemoryRetriever | None = None,
        observer: AgentExecutionObserver | None = None,
    ) -> None:
        self.memory_service = memory_service
        self.memory_retriever = memory_retriever
        self.observer = observer

    async def build(
        self,
        namespace: str,
        *,
        query: str | None = None,
        working_limit: int = 5,
        semantic_limit: int = 5,
        episodic_limit: int = 5,
        agent_name: str | None = None,
        run_id: str | None = None,
    ) -> MemoryContext:
        if not namespace.strip():
            raise ValueError("Memory namespace must not be empty.")

        if working_limit <= 0:
            raise ValueError("working_limit must be greater than zero.")

        if semantic_limit <= 0:
            raise ValueError("semantic_limit must be greater than zero.")

        if episodic_limit <= 0:
            raise ValueError("episodic_limit must be greater than zero.")

        if query is not None and not query.strip():
            raise ValueError("Memory query must not be empty.")

        working = await self.memory_service.recall(
            namespace,
            memory_type="working",
            limit=working_limit,
        )

        semantic_results: tuple[MemoryRetrievalResult, ...] = ()
        episodic_results: tuple[MemoryRetrievalResult, ...] = ()

        if query is not None and self.memory_retriever is not None:
            semantic_results = tuple(
                await self._retrieve_with_observability(
                    query,
                    namespace=namespace,
                    memory_type="semantic",
                    top_k=semantic_limit,
                    agent_name=agent_name,
                    run_id=run_id,
                )
            )
            episodic_results = tuple(
                await self._retrieve_with_observability(
                    query,
                    namespace=namespace,
                    memory_type="episodic",
                    top_k=episodic_limit,
                    agent_name=agent_name,
                    run_id=run_id,
                )
            )

            semantic = tuple(result.item for result in semantic_results)
            episodic = tuple(result.item for result in episodic_results)
        else:
            semantic = await self.memory_service.recall(
                namespace,
                memory_type="semantic",
                limit=semantic_limit,
            )

            episodic = await self.memory_service.recall(
                namespace,
                memory_type="episodic",
                limit=episodic_limit,
            )

        return MemoryContext(
            working=tuple(working),
            semantic=tuple(semantic),
            episodic=tuple(episodic),
            semantic_results=semantic_results,
            episodic_results=episodic_results,
        )

    async def _retrieve_with_observability(
        self,
        query: str,
        *,
        namespace: str,
        memory_type: str,
        top_k: int,
        agent_name: str | None,
        run_id: str | None,
    ) -> tuple[MemoryRetrievalResult, ...]:
        if self.memory_retriever is None:
            raise RuntimeError("Memory retriever is not configured.")

        should_observe = self.observer is not None and agent_name is not None

        base_metadata = {
            "memory_type": memory_type,
            "requested_top_k": top_k,
        }

        if should_observe:
            await self.observer.record(
                AgentExecutionEvent(
                    event_type=AgentExecutionEventType.MEMORY_RETRIEVAL_STARTED,
                    agent_name=agent_name,
                    run_id=run_id,
                    metadata=base_metadata,
                )
            )

        started_at = perf_counter()

        try:
            results = tuple(
                await self.memory_retriever.retrieve(
                    query,
                    namespace=namespace,
                    memory_type=memory_type,
                    top_k=top_k,
                )
            )
        except Exception as exc:
            if should_observe:
                await self.observer.record(
                    AgentExecutionEvent(
                        event_type=AgentExecutionEventType.MEMORY_RETRIEVAL_FAILED,
                        agent_name=agent_name,
                        run_id=run_id,
                        metadata={
                            **base_metadata,
                            "latency_ms": round(
                                (perf_counter() - started_at) * 1000,
                                3,
                            ),
                            "error_type": type(exc).__name__,
                        },
                    )
                )
            raise

        latency_ms = round(
            (perf_counter() - started_at) * 1000,
            3,
        )

        completed_metadata = {
            **base_metadata,
            "returned_count": len(results),
            "latency_ms": latency_ms,
        }

        methods = sorted({result.retrieval_method for result in results if result.retrieval_method})

        if methods:
            completed_metadata["retrieval_methods"] = methods

        retrieval_scores = [
            result.retrieval_score for result in results if result.retrieval_score is not None
        ]

        reranker_scores = [
            result.reranker_score for result in results if result.reranker_score is not None
        ]

        if retrieval_scores:
            completed_metadata["retrieval_score_min"] = min(retrieval_scores)
            completed_metadata["retrieval_score_max"] = max(retrieval_scores)

        if reranker_scores:
            completed_metadata["reranker_score_min"] = min(reranker_scores)
            completed_metadata["reranker_score_max"] = max(reranker_scores)

        if should_observe:
            await self.observer.record(
                AgentExecutionEvent(
                    event_type=AgentExecutionEventType.MEMORY_RETRIEVAL_COMPLETED,
                    agent_name=agent_name,
                    run_id=run_id,
                    metadata=completed_metadata,
                )
            )

        return results
