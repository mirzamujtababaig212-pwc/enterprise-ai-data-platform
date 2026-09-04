from __future__ import annotations

from ai_platform.agents.llm_context import AgentLLMContext
from ai_platform.agents.llm_messages import (
    AgentMessage,
    tool_result_message,
)
from ai_platform.agents.tool_calls import (
    AgentToolCall,
    AgentToolResult,
)
from ai_platform.agents.models import AgentRequest
from ai_platform.agents.tool_context import AgentToolContext
from tools.models import ToolExecutionResult


class AgentExecutionContext:
    """
    Runtime context supplied to an executable agent.

    The context combines the incoming request with the capabilities
    and conversation state available to the agent.

    Agent implementations should use this context rather than
    reaching directly into registries or platform services.
    """

    def __init__(
        self,
        request: AgentRequest,
        *,
        tools: AgentToolContext,
        llm: AgentLLMContext,
        history: tuple[AgentMessage, ...] = (),
    ) -> None:
        self.request = request
        self.tools = tools
        self.llm = llm
        self.history = history

        for message in self.history:
            if not isinstance(message, AgentMessage):
                raise TypeError("Agent execution history must contain " "AgentMessage instances.")

    def build_llm_messages(
        self,
        *,
        tool_results: tuple[str, ...] = (),
    ) -> tuple[AgentMessage, ...]:
        """
        Build the canonical LLM conversation for this execution.

        The bound agent system prompt is followed by conversation
        history, the current user request, and any supplied tool
        results.
        """
        return self.llm.build_messages(
            prompt=self.request.input,
            history=self.history,
            tool_results=tool_results,
        )

    async def build_tool_result_messages(
        self,
        tool_results: tuple[AgentToolResult, ...],
    ) -> tuple[AgentMessage, ...]:
        """
        Convert executed tool results into provider-neutral LLM messages.
        """

        for result in tool_results:
            if not isinstance(result, AgentToolResult):
                raise TypeError("Tool results must contain AgentToolResult instances.")

        return tuple(
            tool_result_message(
                call_id=result.call_id,
                tool_name=result.tool_name,
                output=result.output,
                error=result.error,
            )
            for result in tool_results
        )

    @property
    def session_id(self) -> str | None:
        return self.request.session_id

    @property
    def user_id(self) -> str | None:
        return self.request.user_id

    async def execute_tool_calls(
        self,
        tool_calls: tuple[AgentToolCall, ...],
    ) -> tuple[AgentToolResult, ...]:
        """
        Execute LLM-requested tool calls through the agent tool context.

        Tool authorization and execution remain owned by AgentToolContext.
        This method only coordinates the calls and maps their results into
        the provider-neutral AgentToolResult contract.
        """
        for tool_call in tool_calls:
            if not isinstance(tool_call, AgentToolCall):
                raise TypeError("Tool calls must contain AgentToolCall instances.")

        results: list[AgentToolResult] = []

        for tool_call in tool_calls:
            result = await self.tools.execute(
                tool_call.name,
                tool_call.arguments,
                principal=self.user_id,
            )

            if isinstance(result, ToolExecutionResult):
                results.append(
                    AgentToolResult(
                        call_id=tool_call.call_id,
                        tool_name=tool_call.name,
                        output=result.output if result.success else None,
                        error=result.error if not result.success else None,
                    )
                )
                continue

            if isinstance(result, AgentToolResult):
                results.append(
                    AgentToolResult(
                        call_id=tool_call.call_id,
                        tool_name=tool_call.name,
                        output=result.output,
                        error=result.error,
                    )
                )
                continue

            if isinstance(result, dict):
                success = result.get("success", False)

                if success:
                    results.append(
                        AgentToolResult(
                            call_id=tool_call.call_id,
                            tool_name=tool_call.name,
                            output=result.get("output"),
                        )
                    )
                else:
                    error = result.get("error")

                    if not isinstance(error, str) or not error.strip():
                        error = "Tool execution failed."

                    results.append(
                        AgentToolResult(
                            call_id=tool_call.call_id,
                            tool_name=tool_call.name,
                            error=error,
                        )
                    )

                continue

            results.append(
                AgentToolResult(
                    call_id=tool_call.call_id,
                    tool_name=tool_call.name,
                    error="Tool execution returned an invalid result.",
                )
            )

        return tuple(results)
