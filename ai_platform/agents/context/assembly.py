from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from ai_platform.agents.execution import AgentExecutionContext
from ai_platform.agents.lifecycle import AgentExecutionLifecycleState
from ai_platform.agents.llm_context import AgentLLMContext
from ai_platform.agents.llm_messages import AgentMessage
from ai_platform.agents.models import AgentRequest
from ai_platform.agents.plan_provider import AgentPlanProvider
from ai_platform.agents.decision_provider import AgentRuntimeDecisionProvider
from ai_platform.agents.policy import OutputGovernanceDecision
from ai_platform.agents.tool_context import AgentToolContext
from memory.context.builder import MemoryContext


@dataclass(frozen=True)
class ContextSource:
    """Bounded provenance for one logical context source."""

    source_type: str
    item_id: str | None = None
    score: float | None = None
    reranker_score: float | None = None
    provenance_metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ContextAssemblyResult:
    """Provider-neutral description of an assembled LLM context."""

    messages: tuple[AgentMessage, ...]
    sources: tuple[ContextSource, ...]
    diagnostics: Mapping[str, object]


class AgentContextAssembly:
    """Construct and describe provider-neutral agent execution context.

    This boundary assembles already-resolved runtime capabilities and state.
    It does not retrieve memory, execute tools, call the LLM, authorize access,
    or manage durable execution state.
    """

    @staticmethod
    def assemble(
        request: AgentRequest,
        *,
        tools: AgentToolContext,
        llm: AgentLLMContext,
        history: tuple[AgentMessage, ...] = (),
        memory: MemoryContext | None = None,
        run_id: str | None = None,
        lease_id: str | None = None,
        execution_ownership_lost: asyncio.Event | None = None,
        cancellation_requested: asyncio.Event | None = None,
        output_evaluator: Callable[[str], OutputGovernanceDecision] | None = None,
        agent_run_steps_repository_factory=None,
        lifecycle_state: AgentExecutionLifecycleState | None = None,
        plan_provider: AgentPlanProvider | None = None,
        decision_provider: AgentRuntimeDecisionProvider | None = None,
    ) -> AgentExecutionContext:
        """Build an AgentExecutionContext without changing its semantics."""

        return AgentExecutionContext(
            request,
            output_evaluator=output_evaluator,
            tools=tools,
            llm=llm,
            history=history,
            memory=memory,
            run_id=run_id,
            lease_id=lease_id,
            execution_ownership_lost=execution_ownership_lost,
            cancellation_requested=cancellation_requested,
            agent_run_steps_repository_factory=agent_run_steps_repository_factory,
            lifecycle_state=lifecycle_state,
            plan_provider=plan_provider,
            decision_provider=decision_provider,
        )

    @staticmethod
    def describe(
        context: AgentExecutionContext,
        messages: tuple[AgentMessage, ...],
    ) -> ContextAssemblyResult:
        """Describe the provenance of an already assembled LLM context.

        This method never re-renders messages and never records message content.
        The supplied message tuple remains authoritative for the actual prompt.
        """

        if not isinstance(context, AgentExecutionContext):
            raise TypeError("Context must be an AgentExecutionContext.")

        if not isinstance(messages, tuple):
            raise TypeError("Context messages must be a tuple of AgentMessage instances.")

        if any(not isinstance(message, AgentMessage) for message in messages):
            raise TypeError("Context messages must contain AgentMessage instances.")

        sources: list[ContextSource] = []

        sources.append(
            ContextSource(
                source_type="system_prompt",
            )
        )

        memory = context.memory
        if memory is not None:
            sources.extend(
                AgentContextAssembly._memory_sources(
                    memory.working,
                    source_type="working_memory",
                )
            )
            sources.extend(
                AgentContextAssembly._retrieval_sources(
                    memory.semantic_results,
                    source_type="semantic_memory",
                )
            )
            sources.extend(
                AgentContextAssembly._retrieval_sources(
                    memory.episodic_results,
                    source_type="episodic_memory",
                )
            )

            # Semantic/episodic memory may be supplied without retrieval
            # metadata. Preserve those items as provenance sources rather than
            # inventing retrieval scores.
            sources.extend(
                AgentContextAssembly._unmatched_memory_sources(
                    memory.semantic,
                    existing=sources,
                    source_type="semantic_memory",
                )
            )
            sources.extend(
                AgentContextAssembly._unmatched_memory_sources(
                    memory.episodic,
                    existing=sources,
                    source_type="episodic_memory",
                )
            )

        if context.history:
            sources.append(
                ContextSource(
                    source_type="chat_history",
                    provenance_metadata={
                        "message_count": len(context.history),
                    },
                )
            )

        sources.append(
            ContextSource(
                source_type="user_input",
            )
        )

        tool_result_count = sum(1 for message in messages if message.role == "tool")

        if tool_result_count:
            sources.append(
                ContextSource(
                    source_type="tool_result",
                    provenance_metadata={
                        "result_count": tool_result_count,
                    },
                )
            )

        source_counts: dict[str, int] = {}
        for source in sources:
            source_counts[source.source_type] = source_counts.get(source.source_type, 0) + 1

        diagnostics = {
            "total_messages": len(messages),
            "source_counts": source_counts,
        }

        return ContextAssemblyResult(
            messages=messages,
            sources=tuple(sources),
            diagnostics=diagnostics,
        )

    @staticmethod
    def _memory_sources(
        items: tuple,
        *,
        source_type: str,
    ) -> list[ContextSource]:
        return [
            ContextSource(
                source_type=source_type,
                item_id=item.id,
            )
            for item in items
        ]

    @staticmethod
    def _retrieval_sources(
        results: tuple,
        *,
        source_type: str,
    ) -> list[ContextSource]:
        return [
            ContextSource(
                source_type=source_type,
                item_id=result.item.id,
                score=result.retrieval_score,
                reranker_score=result.reranker_score,
                provenance_metadata={
                    "retrieval_method": result.retrieval_method,
                    "rank": result.rank,
                },
            )
            for result in results
        ]

    @staticmethod
    def _unmatched_memory_sources(
        items: tuple,
        *,
        existing: list[ContextSource],
        source_type: str,
    ) -> list[ContextSource]:
        existing_ids = {
            source.item_id
            for source in existing
            if source.source_type == source_type and source.item_id is not None
        }

        return [
            ContextSource(
                source_type=source_type,
                item_id=item.id,
            )
            for item in items
            if item.id not in existing_ids
        ]
