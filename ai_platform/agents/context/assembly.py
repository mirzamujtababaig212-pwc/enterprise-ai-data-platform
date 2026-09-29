from __future__ import annotations

import asyncio
from collections.abc import Callable

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


class AgentContextAssembly:
    """Construct the provider-neutral execution context for an agent run.

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
