from __future__ import annotations

from typing import Any

from rag.contracts import Retriever
from rag.governance import GovernancePolicy
from tools.models import ToolDefinition, ToolProvider
from tools.execution.context import ToolExecutionContext
from ai_platform.agents.observability import (
    AgentExecutionEvent,
    AgentExecutionEventType,
)
from ai_platform.agents.observer import AgentExecutionObserver
from ai_platform.agents.policy import TenantPolicyEngine


class RAGSearchTool:
    """Tool exposing semantic retrieval to agents."""

    def __init__(
        self,
        retriever: Retriever,
        *,
        observer: AgentExecutionObserver | None = None,
        tenant_policy_engine: TenantPolicyEngine | None = None,
    ) -> None:
        self._retriever = retriever
        self._observer = observer
        self._tenant_policy_engine = tenant_policy_engine

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="rag.search",
            description=(
                "Search the enterprise knowledge base for relevant "
                "documents and return the most relevant sources."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language search query.",
                    },
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "default": 5,
                        "description": "Maximum number of sources to return.",
                    },
                    "min_score": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": (
                            "Optional minimum relevance score. Results below "
                            "this threshold are excluded."
                        ),
                    },
                    "metadata_filter": {
                        "type": "object",
                        "additionalProperties": True,
                        "description": (
                            "Optional equality-based metadata filters. "
                            "Multiple fields are combined with AND semantics."
                        ),
                    },
                },
                "required": ["query"],
            },
            provider=ToolProvider(
                kind="native",
                name="internal",
            ),
            metadata={
                "category": "retrieval",
                "read_only": True,
            },
        )

    async def execute(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        result = await self._execute(
            arguments,
            governance_policy=None,
        )

        await self._emit_rag_governance_decision(
            context=None,
            retrieved_count=result["retrieved_count"],
        )

        return result

    async def _execute(
        self,
        arguments: dict[str, Any],
        *,
        governance_policy: GovernancePolicy | None,
    ) -> dict[str, Any]:
        query = arguments.get("query")

        if not isinstance(query, str) or not query.strip():
            raise ValueError("rag.search query must be a non-empty string.")

        top_k = arguments.get("top_k", 5)

        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise TypeError("rag.search top_k must be an integer.")

        if top_k < 1 or top_k > 10:
            raise ValueError("rag.search top_k must be between 1 and 10.")

        min_score = arguments.get("min_score")

        if min_score is not None:
            if isinstance(min_score, bool) or not isinstance(min_score, (int, float)):
                raise TypeError("rag.search min_score must be a number.")

            if min_score < 0.0 or min_score > 1.0:
                raise ValueError("rag.search min_score must be between 0.0 and 1.0.")

        metadata_filter = arguments.get("metadata_filter")

        if metadata_filter is not None and not isinstance(metadata_filter, dict):
            raise TypeError("rag.search metadata_filter must be an object.")

        results = await self._retriever.retrieve(
            query=query.strip(),
            top_k=top_k,
            min_score=min_score,
            metadata_filter=metadata_filter,
            governance_policy=governance_policy,
        )

        return {
            "query": query.strip(),
            "results": [
                {
                    "chunk_id": result.chunk.id,
                    "document_id": result.chunk.document_id,
                    "content": result.chunk.content,
                    # Compatibility field: final ranking score.
                    "score": result.score,
                    # Canonical evaluation signals.
                    "retrieval_score": (
                        result.retrieval_score
                        if result.retrieval_score is not None
                        else result.score
                    ),
                    "reranker_score": result.reranker_score,
                    "metadata": result.chunk.metadata,
                }
                for result in results
            ],
            "retrieved_count": len(results),
        }

    async def execute_with_context(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        result = await self._execute(
            arguments,
            governance_policy=context.governance_policy,
        )

        await self._emit_rag_governance_decision(
            context=context,
            retrieved_count=result["retrieved_count"],
        )

        return result

    async def _emit_rag_governance_decision(
        self,
        *,
        context: ToolExecutionContext | None,
        retrieved_count: int,
    ) -> None:
        if self._observer is None:
            return

        tenant_id = context.tenant_id if context is not None else None

        metadata: dict[str, object] = {
            "governance_domain": "rag",
            "decision": "allow",
            "retrieved_count": retrieved_count,
        }

        if tenant_id is not None:
            metadata["tenant_id"] = tenant_id

            if self._tenant_policy_engine is not None:
                try:
                    policy = self._tenant_policy_engine.get_policy(tenant_id)
                except Exception:
                    policy = None

                if policy is not None:
                    if policy.policy_id is not None:
                        metadata["policy_id"] = policy.policy_id
                    if policy.policy_version is not None:
                        metadata["policy_version"] = policy.policy_version

        try:
            await self._observer.record(
                AgentExecutionEvent(
                    event_type=AgentExecutionEventType.GOVERNANCE_DECISION,
                    agent_name="rag.search",
                    run_id=context.run_id if context is not None else None,
                    session_id=context.session_id if context is not None else None,
                    user_id=context.user_id if context is not None else None,
                    metadata=metadata,
                )
            )
        except Exception:
            # Governance observability is best-effort.
            return
