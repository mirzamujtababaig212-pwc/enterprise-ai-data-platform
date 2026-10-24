from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ai_platform.agents.observability import (
    AgentExecutionEvent,
    AgentExecutionEventType,
)
from ai_platform.agents.observer import AgentExecutionObserver
from ai_platform.agents.policy import TenantPolicyEngine
from rag.contracts import Retriever
from rag.generation.gateway import GatewayChatService
from rag.governance import GovernancePolicy
from rag.models import RetrievalResult


@dataclass(frozen=True)
class RAGSource:
    """
    Structured evidence exposed to callers of the RAG system.

    The existing source fields remain backward compatible while adding
    stable evidence identity, retrieval ordering, canonical source
    identity, and an optional source locator.
    """

    evidence_id: str
    chunk_id: str
    document_id: str
    score: float
    retrieval_rank: int
    content: str
    metadata: dict[str, Any]
    retrieval_score: float | None = None
    reranker_score: float | None = None
    retrieval_method: str | None = None
    source_ref: dict[str, Any] | None = None
    locator: dict[str, Any] | None = None


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

        if not results:
            return RAGQueryResult(
                answer=(
                    "I couldn't find sufficient evidence in the indexed "
                    "sources to answer this question."
                ),
                sources=(),
                retrieved_count=0,
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

        answer = str(response.get("reply", "")).strip()

        sources = tuple(
            self._to_source(
                result,
                retrieval_rank=index,
            )
            for index, result in enumerate(results, start=1)
        )

        if not self._is_explicit_abstention(answer) and not self._has_valid_citations(
            answer, source_count=len(sources)
        ):
            answer = (
                "I couldn't find sufficient evidence in the indexed "
                "sources to answer this question."
            )

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
            evidence_id = f"evidence:{result.chunk.id}"
            source = result.chunk.metadata.get("source")
            context_blocks.append(
                "\n".join(
                    [
                        f"[Source {index}]",
                        f"Evidence ID: {evidence_id}",
                        f"Document ID: {result.chunk.document_id}",
                        f"Chunk ID: {result.chunk.id}",
                        f"Source Reference: {source}",
                        f"Locator: {result.chunk.metadata.get('locator')}",
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
            "Answer the user's question using only facts supported by "
            "the retrieved context.\n"
            "Cite sources inline using the exact format [Source N], where N "
            "is the source number shown in the context. Include at least one "
            "valid source citation in every non-abstaining answer.\n"
            "Never cite a source number that was not provided in the context.\n"
            "Do not invent facts that are not supported by the context.\n"
            "If the retrieved context does not contain enough information "
            "to answer the question, say so clearly rather than guessing.\n\n"
            f"Retrieved Context:\n{context}\n\n"
            f"User Question:\n{query}\n\n"
            "Answer:"
        )

    @staticmethod
    def _is_explicit_abstention(answer: str) -> bool:
        """Recognize standalone abstention responses without factual claims."""
        normalized = " ".join(answer.split())
        patterns = (
            r"I (?:couldn't|could not) find sufficient evidence"
            r"(?: in the (?:retrieved|indexed) (?:context|sources))?"
            r"(?: to answer this question)?[.!]?",
            r"I (?:don't|do not) have enough information"
            r"(?: in the retrieved sources)?"
            r"(?: to answer this question)?[.!]?",
            r"The retrieved (?:context|sources) "
            r"(?:doesn't|does not|don't|do not) contain enough information"
            r"(?: to answer this question)?[.!]?",
        )
        return any(re.fullmatch(pattern, normalized, flags=re.IGNORECASE) for pattern in patterns)

    @staticmethod
    def _has_valid_citations(answer: str, *, source_count: int) -> bool:
        """Return whether an answer cites only valid, retrieved source numbers."""
        markers = re.findall(r"\[Source[^\]]*(?:\]|$)", answer, flags=re.IGNORECASE)
        if not markers:
            return False

        for marker in markers:
            match = re.fullmatch(r"\[Source\s+(\d+)\]", marker, flags=re.IGNORECASE)
            if match is None:
                return False

            source_number = int(match.group(1))
            if not 1 <= source_number <= source_count:
                return False

        return True

    @staticmethod
    def _to_source(
        result: RetrievalResult,
        *,
        retrieval_rank: int,
    ) -> RAGSource:
        metadata = dict(result.chunk.metadata)

        source_ref = metadata.get("source_ref")
        if not isinstance(source_ref, dict):
            source = metadata.get("source")
            source_ref = {"source": source} if isinstance(source, str) and source.strip() else None

        locator = metadata.get("locator")
        if not isinstance(locator, dict):
            locator = None

        retrieval_method = metadata.get("retrieval_method")
        if not isinstance(retrieval_method, str) or not retrieval_method.strip():
            retrieval_method = None

        return RAGSource(
            evidence_id=f"evidence:{result.chunk.id}",
            chunk_id=result.chunk.id,
            document_id=result.chunk.document_id,
            score=result.score,
            retrieval_rank=retrieval_rank,
            content=result.chunk.content,
            metadata=metadata,
            retrieval_score=result.retrieval_score,
            reranker_score=result.reranker_score,
            retrieval_method=retrieval_method,
            source_ref=dict(source_ref) if source_ref is not None else None,
            locator=dict(locator) if locator is not None else None,
        )
