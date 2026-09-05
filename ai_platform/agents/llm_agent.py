from __future__ import annotations

from ai_platform.agents.observer import AgentExecutionObserver
from ai_platform.agents.exceptions import AgentToolLoopLimitError
from ai_platform.agents.execution import AgentExecutionContext
from ai_platform.agents.llm_messages import AgentMessage
from ai_platform.agents.models import (
    AgentDefinition,
    AgentResponse,
)
from ai_platform.agents.observability import (
    AgentExecutionEvent,
    AgentExecutionEventType,
)


class LLMAgent:
    """
    Concrete LLM-backed enterprise AI agent.

    This implementation owns the agent-level interaction with the
    AgentLLMContext. Tool-call orchestration is added incrementally
    while the AgentExecutionContext remains responsible for tool
    execution and provider-neutral message construction.
    """

    MAX_TOOL_ROUNDS = 3

    def __init__(
        self,
        definition: AgentDefinition,
        *,
        observer: AgentExecutionObserver | None = None,
    ) -> None:
        if not isinstance(definition, AgentDefinition):
            raise TypeError("LLMAgent definition must be an AgentDefinition.")

        self._definition = definition
        self._observer = observer

    @property
    def definition(self) -> AgentDefinition:
        """
        Return the static definition of this agent.
        """
        return self._definition

    async def _emit(
        self,
        event: AgentExecutionEvent,
    ) -> None:
        """
        Emit an execution event when an observer is configured.

        Agent execution must remain functional when no observer is
        configured.
        """
        if self._observer is None:
            return

        await self._observer.record(event)

    async def _accumulate_tool_call_messages(
        self,
        messages: list[AgentMessage],
        context: AgentExecutionContext,
        tool_calls,
        *,
        tool_round: int,
        assistant_content: str = "",
    ) -> None:
        """
        Append the assistant tool-call message and corresponding tool
        result messages to the current conversation.

        Tool execution remains owned by AgentExecutionContext.
        This method only coordinates execution results into the
        provider-neutral conversation representation and emits
        provider-neutral tool-call lifecycle events.
        """
        from ai_platform.agents.llm_messages import (
            assistant_tool_call_message,
        )

        messages.append(
            assistant_tool_call_message(
                tool_calls=tool_calls,
                content=assistant_content,
            )
        )

        for tool_call in tool_calls:
            await self._emit(
                AgentExecutionEvent(
                    event_type=AgentExecutionEventType.TOOL_CALL_REQUESTED,
                    agent_name=self.definition.name,
                    session_id=context.session_id,
                    tool_round=tool_round,
                    tool_name=tool_call.name,
                    call_id=tool_call.call_id,
                )
            )

        tool_results = await context.execute_tool_calls(
            tool_calls,
        )

        for tool_call, tool_result in zip(
            tool_calls,
            tool_results,
        ):
            if tool_result.success:
                await self._emit(
                    AgentExecutionEvent(
                        event_type=AgentExecutionEventType.TOOL_CALL_COMPLETED,
                        agent_name=self.definition.name,
                        session_id=context.session_id,
                        tool_round=tool_round,
                        tool_name=tool_call.name,
                        call_id=tool_call.call_id,
                    )
                )
            else:
                await self._emit(
                    AgentExecutionEvent(
                        event_type=AgentExecutionEventType.TOOL_CALL_FAILED,
                        agent_name=self.definition.name,
                        session_id=context.session_id,
                        tool_round=tool_round,
                        tool_name=tool_call.name,
                        call_id=tool_call.call_id,
                    )
                )

        tool_result_messages = await context.build_tool_result_messages(
            tool_results,
        )

        messages.extend(
            tool_result_messages,
        )

    async def run(
        self,
        context: AgentExecutionContext,
    ) -> AgentResponse:
        """
        Execute the agent interaction.

        The agent emits provider-neutral lifecycle events when an
        execution observer is configured.
        """
        if not isinstance(context, AgentExecutionContext):
            raise TypeError("LLMAgent context must be an AgentExecutionContext.")

        await self._emit(
            AgentExecutionEvent(
                event_type=AgentExecutionEventType.AGENT_STARTED,
                agent_name=self.definition.name,
                session_id=context.session_id,
            )
        )

        try:
            messages = list(context.build_llm_messages())
            tools = await context.tools.list_tools()
            tool_rounds = 0

            while True:
                await self._emit(
                    AgentExecutionEvent(
                        event_type=AgentExecutionEventType.LLM_REQUESTED,
                        agent_name=self.definition.name,
                        session_id=context.session_id,
                        tool_round=tool_rounds,
                    )
                )

                result = await context.llm.generate(
                    prompt=context.request.input,
                    messages=tuple(messages),
                    tools=tuple(tools),
                    user_id=context.user_id,
                    metadata=context.metadata,
                )

                await self._emit(
                    AgentExecutionEvent(
                        event_type=AgentExecutionEventType.LLM_COMPLETED,
                        agent_name=self.definition.name,
                        session_id=context.session_id,
                        tool_round=tool_rounds,
                        provider=result.provider,
                        model=result.model,
                        metadata={
                            "prompt_tokens": result.usage.prompt_tokens,
                            "completion_tokens": result.usage.completion_tokens,
                            "total_tokens": result.usage.total_tokens,
                        },
                    )
                )

                if not result.tool_calls:
                    await self._emit(
                        AgentExecutionEvent(
                            event_type=AgentExecutionEventType.AGENT_COMPLETED,
                            agent_name=self.definition.name,
                            session_id=context.session_id,
                            tool_round=tool_rounds,
                            provider=result.provider,
                            model=result.model,
                        )
                    )

                    return AgentResponse(
                        agent_name=self.definition.name,
                        output=result.text,
                        session_id=context.session_id,
                        metadata={
                            "provider": result.provider,
                            "model": result.model,
                            "usage": {
                                "prompt_tokens": result.usage.prompt_tokens,
                                "completion_tokens": result.usage.completion_tokens,
                                "total_tokens": result.usage.total_tokens,
                            },
                            "tool_rounds": tool_rounds,
                        },
                    )

                if tool_rounds >= self.MAX_TOOL_ROUNDS:
                    raise AgentToolLoopLimitError(
                        self.definition.name,
                        self.MAX_TOOL_ROUNDS,
                    )

                tool_rounds += 1

                await self._accumulate_tool_call_messages(
                    messages,
                    context,
                    result.tool_calls,
                    tool_round=tool_rounds,
                    assistant_content=result.text,
                )

        except Exception as exc:
            await self._emit(
                AgentExecutionEvent(
                    event_type=AgentExecutionEventType.AGENT_FAILED,
                    agent_name=self.definition.name,
                    session_id=context.session_id,
                    tool_round=tool_rounds,
                    metadata={
                        "error_type": type(exc).__name__,
                    },
                )
            )
            raise
