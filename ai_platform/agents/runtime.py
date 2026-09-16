from __future__ import annotations

from ai_platform.agents.contracts import AgentRegistry
from ai_platform.agents.execution import AgentExecutionContext
from ai_platform.agents.llm_context import (
    AgentLLMContext,
    LLMGateway,
    UnavailableLLMGateway,
)
from ai_platform.agents.models import AgentRequest, AgentResponse
from ai_platform.agents.llm_messages import AgentMessage
from ai_platform.agents.tool_context import AgentToolContext
from memory.context.builder import MemoryContextBuilder
from memory.service import MemoryService
from tools.contracts import ToolRegistry
from tools.execution.service import ToolExecutionService


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

    async def run(
        self,
        agent_name: str,
        request: AgentRequest,
        *,
        history: tuple[AgentMessage, ...] = (),
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
            )

        llm_context = AgentLLMContext(
            self._llm_gateway,
            agent.definition.llm_config,
        )

        context = AgentExecutionContext(
            request,
            tools=tool_context,
            llm=llm_context,
            history=history,
            memory=memory_context,
        )

        response = await agent.run(context)

        if (
            agent.definition.memory_write_enabled
            and request.memory_namespace is not None
            and self._memory_service is not None
            and isinstance(response.output, str)
            and response.output.strip()
        ):
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
                    memory_type="episodic",
                    metadata=metadata,
                    retention_seconds=(agent.definition.memory_episodic_retention_seconds),
                )
            except Exception:
                # Memory persistence must not turn a successful agent
                # execution into a failed agent execution.
                pass

        return response
