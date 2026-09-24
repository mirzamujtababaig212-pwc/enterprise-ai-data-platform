from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from rag.contracts import Retriever
from ai_platform.agents.policy import TenantPolicyEngine
from rag.generation.gateway import GatewayChatService
from rag.governance import GovernancePolicy
from rag.models import RetrievalResult
from ai_platform.agents.observability import (
    AgentExecutionEvent,
    AgentExecutionEventType,
)
from ai_platform.agents.observer import AgentExecutionObserver


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
        tenant_policy_engine: TenantPolicyEngine | None = None,
        observer: AgentExecutionObserver | None = None,
    ) -> None:
        self.retriever = retriever
        self.chat_service = chat_service
        self.tenant_policy_engine = tenant_policy_engine
        self.observer = observer

    async def query(
        self,
        query: str,
        *,
        top_k: int = 5,
        min_score: float | None = None,
        metadata_filter: Mapping[str, object] | None = None,
        governance_policy: GovernancePolicy | None = None,
        tenant_id: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        user_id: str | None = None,
    ) -> RAGQueryResult:
        if not query.strip():
            raise ValueError("Query must not be empty.")

        effective_governance_policy = self._resolve_governance_policy(
            tenant_id=tenant_id,
            governance_policy=governance_policy,
            metadata_filter=metadata_filter,
        )

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        retrieval_kwargs = {
            "top_k": top_k,
            "min_score": min_score,
            "metadata_filter": metadata_filter,
        }

        if effective_governance_policy is not None:
            retrieval_kwargs["governance_policy"] = effective_governance_policy

        results = await self.retriever.retrieve(
            query,
            **retrieval_kwargs,
        )

        await self._emit_rag_governance_decision(
            tenant_id=tenant_id,
            retrieved_count=len(results),
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

    async def _emit_rag_governance_decision(
        self,
        *,
        tenant_id: str | None,
        retrieved_count: int,
    ) -> None:
        if self.observer is None:
            return

        metadata: dict[str, object] = {
            "governance_domain": "rag",
            "decision": "allow",
            "retrieved_count": retrieved_count,
        }

        if tenant_id is not None:
            metadata["tenant_id"] = tenant_id

            if self.tenant_policy_engine is not None:
                try:
                    policy = self.tenant_policy_engine.get_policy(tenant_id)
                except Exception:
                    policy = None

                if policy is not None:
                    if policy.policy_id is not None:
                        metadata["policy_id"] = policy.policy_id
                    if policy.policy_version is not None:
                        metadata["policy_version"] = policy.policy_version

        try:
            await self.observer.record(
                AgentExecutionEvent(
                    event_type=AgentExecutionEventType.GOVERNANCE_DECISION,
                    agent_name="rag.query",
                    run_id=None,
                    session_id=None,
                    user_id=None,
                    metadata=metadata,
                )
            )
        except Exception:
            # Governance observability is best-effort.
            return

    def _resolve_governance_policy(
        self,
        *,
        tenant_id: str | None,
        governance_policy: GovernancePolicy | None,
        metadata_filter: Mapping[str, object] | None,
    ) -> GovernancePolicy | None:
        if tenant_id is None:
            return governance_policy

        tenant_policy = (
            self.tenant_policy_engine.get_policy(tenant_id)
            if self.tenant_policy_engine is not None
            else None
        )

        cross_tenant_allowed = tenant_policy is not None and tenant_policy.allow_cross_tenant_data

        if not cross_tenant_allowed:
            requested_tenant_id = (
                metadata_filter.get("tenant_id") if metadata_filter is not None else None
            )

            if requested_tenant_id is not None and requested_tenant_id != tenant_id:
                raise ValueError("Metadata filter tenant_id conflicts with authenticated tenant.")

        if cross_tenant_allowed:
            return governance_policy

        base_policy = governance_policy if governance_policy is not None else GovernancePolicy()

        return base_policy.with_tenant_scope(tenant_id)

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
