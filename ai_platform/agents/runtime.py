from __future__ import annotations

import asyncio
import time

from ai_platform.agents.checkpoint import AgentExecutionCheckpoint
from ai_platform.agents.decision_provider import (
    AgentRuntimeDecisionProvider,
    DeterministicAgentRuntimeDecisionProvider,
)
from ai_platform.agents.plan_provider import (
    AgentPlanProvider,
    DeterministicAgentPlanProvider,
)
from ai_platform.agents.contracts import (
    AgentExecutionContract,
    AgentRegistry,
)
from ai_platform.agents.context.assembly import AgentContextAssembly
from ai_platform.agents.exceptions import AgentExecutionControlSignal
from ai_platform.agents.llm_context import (
    AgentLLMContext,
    LLMGateway,
    UnavailableLLMGateway,
)
from ai_platform.agents.llm_messages import AgentMessage
from ai_platform.agents.lifecycle import (
    AgentExecutionLifecyclePhase,
    AgentExecutionLifecycleState,
)
from ai_platform.agents.models import AgentDefinition, AgentRequest, AgentResponse
from ai_platform.agents.policy import TenantPolicyEngine
from ai_platform.agents.observability import (
    AgentExecutionEvent,
    AgentExecutionEventType,
)
from ai_platform.agents.observer import AgentExecutionObserver
from ai_platform.agents.tool_context import AgentToolContext
from memory.context.builder import MemoryContextBuilder
from memory.service import MemoryService
from tools.contracts import ToolRegistry
from tools.execution.service import ToolExecutionService


async def _record_memory_write_event(
    observer: AgentExecutionObserver | None,
    event: AgentExecutionEvent,
) -> None:
    if observer is None:
        return

    try:
        await observer.record(event)
    except Exception:
        # Observability must not change the outcome of agent execution.
        pass


class AgentRuntime:
    """
    Execution boundary for enterprise AI agents.

    The runtime is responsible for:

    - resolving an agent from the registry
    - validating that the agent is available
    - constructing its execution context
    - exposing declared tool capabilities
    - exposing the existing LLM Gateway
    - invoking the agent
    - returning the agent response

    Provider routing, fallback, retries, authentication, metrics,
    tracing, and provider execution remain owned by the LLM Gateway.
    """

    def __init__(
        self,
        registry: AgentRegistry,
        *,
        tool_registry: ToolRegistry | None = None,
        tool_execution_service: ToolExecutionService | None = None,
        llm_gateway: LLMGateway | None = None,
        memory_context_builder: MemoryContextBuilder | None = None,
        memory_service: MemoryService | None = None,
        observer: AgentExecutionObserver | None = None,
        agent_run_steps_repository_factory=None,
        tenant_policy_engine: TenantPolicyEngine | None = None,
        plan_provider: AgentPlanProvider | None = None,
        decision_provider: AgentRuntimeDecisionProvider | None = None,
    ) -> None:
        self._registry = registry
        self._tool_registry = tool_registry

        if tool_execution_service is not None and tool_registry is None:
            tool_registry = tool_execution_service.registry

        self._tool_execution_service = (
            tool_execution_service
            if tool_execution_service is not None
            else (ToolExecutionService(tool_registry) if tool_registry is not None else None)
        )

        self._tool_registry = tool_registry
        self._llm_gateway = llm_gateway if llm_gateway is not None else UnavailableLLMGateway()
        self._memory_context_builder = memory_context_builder
        self._memory_service = memory_service
        self._observer = observer
        self._agent_run_steps_repository_factory = agent_run_steps_repository_factory
        self._tenant_policy_engine = tenant_policy_engine
        self._plan_provider = plan_provider or DeterministicAgentPlanProvider()
        self._decision_provider = decision_provider or DeterministicAgentRuntimeDecisionProvider()

    async def get_agent_definition(self, agent_name: str) -> AgentDefinition:
        """Retrieve the AgentDefinition for a given agent name from the registry."""
        if not agent_name.strip():
            raise ValueError("Agent name must not be empty.")
        agent = await self._registry.get(agent_name)
        if agent is None:
            raise LookupError(f"Agent '{agent_name}' is not registered.")
        return agent.definition

    async def resume(
        self,
        agent_name: str,
        request: AgentRequest,
        checkpoint: AgentExecutionCheckpoint,
        *,
        run_id: str | None = None,
        lease_id: str | None = None,
        execution_ownership_lost: asyncio.Event | None = None,
        cancellation_requested: asyncio.Event | None = None,
    ) -> AgentResponse:
        """
        Resume a recoverable agent from a durable execution checkpoint.

        The checkpoint conversation is authoritative continuation state.
        Memory is intentionally not rebuilt and successful episodic memory
        write-back is intentionally skipped during recovery.
        """
        if not agent_name.strip():
            raise ValueError("Agent name must not be empty.")

        if not isinstance(checkpoint, AgentExecutionCheckpoint):
            raise TypeError("AgentRuntime checkpoint must be an AgentExecutionCheckpoint.")

        agent = await self._registry.get(agent_name)

        if agent is None:
            raise LookupError(f"Agent '{agent_name}' is not registered.")

        if not agent.definition.enabled:
            raise RuntimeError(f"Agent '{agent_name}' is disabled.")

        resume_agent = getattr(agent, "resume", None)

        if resume_agent is None or not callable(resume_agent):
            raise TypeError(f"Agent '{agent_name}' does not support durable recovery.")

        if checkpoint.agent_name != agent_name:
            raise ValueError("Checkpoint agent_name does not match the requested agent.")

        if checkpoint.run_id != run_id:
            raise ValueError("Checkpoint run_id does not match the AgentRuntime run_id.")

        if agent.definition.tool_names:
            if self._tool_registry is None:
                raise RuntimeError(
                    f"Agent '{agent_name}' declares tools but " "no ToolRegistry is configured."
                )

            tool_context = AgentToolContext(
                self._tool_registry,
                agent.definition,
                execution_service=self._tool_execution_service,
            )
        else:
            if self._tool_registry is not None:
                tool_context = AgentToolContext(
                    self._tool_registry,
                    agent.definition,
                    execution_service=self._tool_execution_service,
                )
            else:

                class EmptyToolRegistry:
                    async def get(
                        self,
                        name: str,
                    ):
                        return None

                    async def list_tools(self):
                        return []

                tool_context = AgentToolContext(
                    EmptyToolRegistry(),
                    agent.definition,
                )

        llm_context = AgentLLMContext(
            self._llm_gateway,
            agent.definition.llm_config,
        )

        output_evaluator = None
        if self._tenant_policy_engine is not None:

            def evaluate_output(text: str):
                return self._tenant_policy_engine.evaluate_output(
                    request.tenant_id,
                    text,
                )

            output_evaluator = evaluate_output

        context = AgentContextAssembly.assemble(
            request,
            output_evaluator=output_evaluator,
            tools=tool_context,
            llm=llm_context,
            history=checkpoint.messages,
            memory=None,
            run_id=run_id,
            lease_id=lease_id,
            execution_ownership_lost=execution_ownership_lost,
            cancellation_requested=cancellation_requested,
            agent_run_steps_repository_factory=(self._agent_run_steps_repository_factory),
            plan_provider=self._plan_provider,
            decision_provider=self._decision_provider,
        )

        try:
            context.install_orchestration_plan(context.plan_provider.build_plan(context))
        except LookupError:
            pass

        return await resume_agent(context, checkpoint)

    async def run(
        self,
        agent_name: str,
        request: AgentRequest,
        *,
        history: tuple[AgentMessage, ...] = (),
        run_id: str | None = None,
        lease_id: str | None = None,
        execution_ownership_lost: asyncio.Event | None = None,
        cancellation_requested: asyncio.Event | None = None,
    ) -> AgentResponse:
        if not agent_name.strip():
            raise ValueError("Agent name must not be empty.")

        agent = await self._registry.get(agent_name)

        if agent is None:
            raise LookupError(f"Agent '{agent_name}' is not registered.")

        if not agent.definition.enabled:
            raise RuntimeError(f"Agent '{agent_name}' is disabled.")

        if agent.definition.tool_names:
            if self._tool_registry is None:
                raise RuntimeError(
                    f"Agent '{agent_name}' declares tools but " "no ToolRegistry is configured."
                )

            tool_context = AgentToolContext(
                self._tool_registry,
                agent.definition,
                execution_service=self._tool_execution_service,
            )
        else:
            if self._tool_registry is not None:
                tool_context = AgentToolContext(
                    self._tool_registry,
                    agent.definition,
                    execution_service=self._tool_execution_service,
                )
            else:

                class EmptyToolRegistry:
                    async def get(
                        self,
                        name: str,
                    ):
                        return None

                    async def list_tools(self):
                        return []

                empty_registry = EmptyToolRegistry()

                tool_context = AgentToolContext(
                    empty_registry,
                    agent.definition,
                )

        memory_context = None

        if request.memory_namespace is not None:
            if self._memory_context_builder is None:
                raise RuntimeError(
                    f"Agent '{agent_name}' requested memory but no "
                    "MemoryContextBuilder is configured."
                )

            memory_context = await self._memory_context_builder.build(
                request.memory_namespace,
                query=request.input,
                agent_name=agent_name,
                run_id=run_id,
            )

        llm_context = AgentLLMContext(
            self._llm_gateway,
            agent.definition.llm_config,
        )

        output_evaluator = None
        if self._tenant_policy_engine is not None:

            def evaluate_output(text: str):
                return self._tenant_policy_engine.evaluate_output(
                    request.tenant_id,
                    text,
                )

            output_evaluator = evaluate_output

        lifecycle_state = AgentExecutionLifecycleState()

        context = AgentContextAssembly.assemble(
            request,
            output_evaluator=output_evaluator,
            tools=tool_context,
            llm=llm_context,
            history=history,
            memory=memory_context,
            run_id=run_id,
            lease_id=lease_id,
            execution_ownership_lost=execution_ownership_lost,
            cancellation_requested=cancellation_requested,
            agent_run_steps_repository_factory=(self._agent_run_steps_repository_factory),
            lifecycle_state=lifecycle_state,
            plan_provider=self._plan_provider,
            decision_provider=self._decision_provider,
        )

        try:
            context.install_orchestration_plan(context.plan_provider.build_plan(context))
        except LookupError:
            pass

        try:
            await lifecycle_state.transition(
                AgentExecutionLifecyclePhase.PREPARING,
                expected_phase=AgentExecutionLifecyclePhase.CREATED,
            )

            if isinstance(agent, AgentExecutionContract):
                await agent.prepare_context(context)

                await lifecycle_state.transition(
                    AgentExecutionLifecyclePhase.PRE_EXECUTION,
                    expected_phase=AgentExecutionLifecyclePhase.PREPARING,
                )
                await agent.evaluate_pre_execution(context)

                await lifecycle_state.transition(
                    AgentExecutionLifecyclePhase.ORCHESTRATING,
                    expected_phase=AgentExecutionLifecyclePhase.PRE_EXECUTION,
                )
                await agent.orchestrate_step(context)

                await lifecycle_state.transition(
                    AgentExecutionLifecyclePhase.EXECUTING,
                    expected_phase=AgentExecutionLifecyclePhase.ORCHESTRATING,
                )
                response = await agent.execute_boundary(context)

                await lifecycle_state.transition(
                    AgentExecutionLifecyclePhase.POST_EXECUTION,
                    expected_phase=AgentExecutionLifecyclePhase.EXECUTING,
                )
                response = await agent.evaluate_post_execution(context, response)

                await lifecycle_state.transition(
                    AgentExecutionLifecyclePhase.COMPLETED,
                    expected_phase=AgentExecutionLifecyclePhase.POST_EXECUTION,
                )
            else:
                await lifecycle_state.transition(
                    AgentExecutionLifecyclePhase.PRE_EXECUTION,
                    expected_phase=AgentExecutionLifecyclePhase.PREPARING,
                )
                await lifecycle_state.transition(
                    AgentExecutionLifecyclePhase.ORCHESTRATING,
                    expected_phase=AgentExecutionLifecyclePhase.PRE_EXECUTION,
                )
                await lifecycle_state.transition(
                    AgentExecutionLifecyclePhase.EXECUTING,
                    expected_phase=AgentExecutionLifecyclePhase.ORCHESTRATING,
                )
                response = await agent.run(context)

                await lifecycle_state.transition(
                    AgentExecutionLifecyclePhase.POST_EXECUTION,
                    expected_phase=AgentExecutionLifecyclePhase.EXECUTING,
                )
                await lifecycle_state.transition(
                    AgentExecutionLifecyclePhase.COMPLETED,
                    expected_phase=AgentExecutionLifecyclePhase.POST_EXECUTION,
                )
        except AgentExecutionControlSignal:
            raise
        except Exception:
            if lifecycle_state.phase is not AgentExecutionLifecyclePhase.FAILED:
                await lifecycle_state.transition(
                    AgentExecutionLifecyclePhase.FAILED,
                    expected_phase=lifecycle_state.phase,
                )
            raise

        if (
            agent.definition.memory_write_enabled
            and request.memory_namespace is not None
            and self._memory_service is not None
            and isinstance(response.output, str)
            and response.output.strip()
        ):
            memory_type = "episodic"
            write_started_at = time.perf_counter()

            await _record_memory_write_event(
                self._observer,
                AgentExecutionEvent(
                    event_type=AgentExecutionEventType.MEMORY_WRITE_STARTED,
                    agent_name=agent.definition.name,
                    run_id=run_id,
                    metadata={
                        "memory_type": memory_type,
                    },
                ),
            )

            metadata = {
                "source": "agent_execution",
                "agent_name": agent.definition.name,
            }

            if request.session_id is not None:
                metadata["session_id"] = request.session_id

            try:
                await self._memory_service.remember(
                    response.output,
                    namespace=request.memory_namespace,
                    memory_type=memory_type,
                    metadata=metadata,
                    retention_seconds=(agent.definition.memory_episodic_retention_seconds),
                )
            except Exception as exc:
                latency_ms = (time.perf_counter() - write_started_at) * 1000.0

                await _record_memory_write_event(
                    self._observer,
                    AgentExecutionEvent(
                        event_type=AgentExecutionEventType.MEMORY_WRITE_FAILED,
                        agent_name=agent.definition.name,
                        run_id=run_id,
                        metadata={
                            "memory_type": memory_type,
                            "latency_ms": latency_ms,
                            "error_type": type(exc).__name__,
                        },
                    ),
                )

                # Memory persistence must not turn a successful agent
                # execution into a failed agent execution.
            else:
                latency_ms = (time.perf_counter() - write_started_at) * 1000.0

                await _record_memory_write_event(
                    self._observer,
                    AgentExecutionEvent(
                        event_type=AgentExecutionEventType.MEMORY_WRITE_COMPLETED,
                        agent_name=agent.definition.name,
                        run_id=run_id,
                        metadata={
                            "memory_type": memory_type,
                            "latency_ms": latency_ms,
                        },
                    ),
                )

        return response
