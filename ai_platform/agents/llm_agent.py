from __future__ import annotations

from ai_platform.agents.budget import ExecutionBudgetState
from ai_platform.agents.observer import AgentExecutionObserver
from ai_platform.agents.checkpoint import (
    AgentCheckpointHandler,
    AgentCheckpointPosition,
    AgentExecutionCheckpoint,
)
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
        checkpoint_handler: AgentCheckpointHandler | None = None,
    ) -> None:
        if not isinstance(definition, AgentDefinition):
            raise TypeError("LLMAgent definition must be an AgentDefinition.")

        self._definition = definition
        self._observer = observer
        self._checkpoint_handler = checkpoint_handler

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

    @staticmethod
    def _tool_provenance_metadata(
        tool_name: str,
        output: object,
    ) -> dict[str, object]:
        """
        Extract bounded provenance metadata from supported tool outputs.

        RAG provenance records source identity and relevance only. Tool
        content and arbitrary payloads remain outside execution events.
        """
        if tool_name != "rag.search" or not isinstance(output, dict):
            return {}

        results = output.get("results")
        if not isinstance(results, list):
            return {}

        sources = []
        for result in results:
            if not isinstance(result, dict):
                continue

            chunk_id = result.get("chunk_id")
            document_id = result.get("document_id")
            score = result.get("score")

            if not isinstance(chunk_id, str) or not chunk_id:
                continue
            if not isinstance(document_id, str) or not document_id:
                continue
            if not isinstance(score, (int, float)) or isinstance(score, bool):
                continue

            sources.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "score": score,
                }
            )

        if not sources:
            return {}

        retrieved_count = output.get("retrieved_count")
        if not isinstance(retrieved_count, int) or isinstance(retrieved_count, bool):
            retrieved_count = len(sources)

        return {
            "rag_provenance": {
                "retrieved_count": retrieved_count,
                "sources": sources,
            }
        }

    async def _execute_tool_calls_and_append_results(
        self,
        messages: list[AgentMessage],
        context: AgentExecutionContext,
        tool_calls,
        *,
        tool_round: int,
    ) -> None:
        """
        Execute already-captured tool calls and append their results.

        This helper is shared by normal execution and checkpoint recovery.
        It deliberately does not consume execution budget because budget
        consumption happens when the LLM originally produced the tool calls.
        """
        context.raise_if_execution_ownership_lost()

        for tool_call in tool_calls:
            context.raise_if_execution_ownership_lost()
            await self._emit(
                AgentExecutionEvent(
                    event_type=AgentExecutionEventType.TOOL_CALL_REQUESTED,
                    agent_name=self.definition.name,
                    run_id=context.run_id,
                    session_id=context.session_id,
                    user_id=context.user_id,
                    tool_round=tool_round,
                    tool_name=tool_call.name,
                    call_id=tool_call.call_id,
                )
            )

        context.raise_if_execution_ownership_lost()

        tool_results = await context.execute_tool_calls(
            tool_calls,
        )

        context.raise_if_execution_ownership_lost()

        for tool_call, tool_result in zip(
            tool_calls,
            tool_results,
        ):
            if tool_result.success:
                await self._emit(
                    AgentExecutionEvent(
                        event_type=AgentExecutionEventType.TOOL_CALL_COMPLETED,
                        agent_name=self.definition.name,
                        run_id=context.run_id,
                        session_id=context.session_id,
                        user_id=context.user_id,
                        tool_round=tool_round,
                        tool_name=tool_call.name,
                        call_id=tool_call.call_id,
                        metadata=self._tool_provenance_metadata(
                            tool_call.name,
                            tool_result.output,
                        ),
                    )
                )
            else:
                await self._emit(
                    AgentExecutionEvent(
                        event_type=AgentExecutionEventType.TOOL_CALL_FAILED,
                        agent_name=self.definition.name,
                        run_id=context.run_id,
                        session_id=context.session_id,
                        user_id=context.user_id,
                        tool_round=tool_round,
                        tool_name=tool_call.name,
                        call_id=tool_call.call_id,
                        metadata=(
                            {
                                "failure_category": tool_result.failure_category.value,
                            }
                            if tool_result.failure_category is not None
                            else {}
                        ),
                    )
                )

        tool_result_messages = await context.build_tool_result_messages(
            tool_results,
        )

        messages.extend(
            tool_result_messages,
        )

        context.raise_if_execution_ownership_lost()

        if self._checkpoint_handler is not None and context.run_id is not None:
            context.raise_if_execution_ownership_lost()
            await self._checkpoint_handler.save(
                AgentExecutionCheckpoint(
                    schema_version=AgentExecutionCheckpoint.CURRENT_SCHEMA_VERSION,
                    run_id=context.run_id,
                    agent_name=self.definition.name,
                    session_id=context.session_id,
                    user_id=context.user_id,
                    messages=tuple(messages),
                    tool_round=tool_round,
                    position=AgentCheckpointPosition.AFTER_TOOL_EXECUTION,
                    metadata=context.metadata,
                    execution_budget_state=ExecutionBudgetState.from_dict(
                        context.execution_budget_state.to_dict()
                    ),
                ),
                lease_id=context.lease_id,
            )

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
        Append the assistant tool-call message and execute its tool calls.
        """
        from ai_platform.agents.llm_messages import (
            assistant_tool_call_message,
        )

        context.raise_if_execution_ownership_lost()

        messages.append(
            assistant_tool_call_message(
                tool_calls=tool_calls,
                content=assistant_content,
            )
        )

        context.raise_if_execution_ownership_lost()

        if self._checkpoint_handler is not None and context.run_id is not None:
            context.raise_if_execution_ownership_lost()
            await self._checkpoint_handler.save(
                AgentExecutionCheckpoint(
                    schema_version=AgentExecutionCheckpoint.CURRENT_SCHEMA_VERSION,
                    run_id=context.run_id,
                    agent_name=self.definition.name,
                    session_id=context.session_id,
                    user_id=context.user_id,
                    messages=tuple(messages),
                    tool_round=tool_round,
                    position=AgentCheckpointPosition.BEFORE_TOOL_EXECUTION,
                    metadata=context.metadata,
                    execution_budget_state=ExecutionBudgetState.from_dict(
                        context.execution_budget_state.to_dict()
                    ),
                ),
                lease_id=context.lease_id,
            )

        context.raise_if_execution_ownership_lost()

        await self._execute_tool_calls_and_append_results(
            messages,
            context,
            tool_calls,
            tool_round=tool_round,
        )

    def _start_orchestration_step(
        self,
        context: AgentExecutionContext,
    ) -> int | None:
        state = context.orchestration_state

        if context.orchestration_plan is None:
            return None

        step = state.current_step
        if step is None:
            step = state.start_step(0)

        return step.step_index

    def _complete_orchestration_step(
        self,
        context: AgentExecutionContext,
        step_index: int | None,
        *,
        tool_round: int | None = None,
    ) -> None:
        if step_index is None:
            return

        state = context.orchestration_state
        state.complete_step(
            step_index,
            tool_round=tool_round,
        )

    async def _continue(
        self,
        context: AgentExecutionContext,
        messages: list[AgentMessage],
        *,
        tool_rounds: int,
    ) -> AgentResponse:
        """
        Continue an agent execution from an existing conversation state.

        ``messages`` is authoritative continuation state. In particular,
        a resumed execution must not rebuild the conversation from the
        original request, history, or memory because the checkpoint may
        already contain completed tool results.
        """
        try:
            tools = await context.tools.list_tools()

            while True:
                context.raise_if_execution_ownership_lost()

                context.execution_budget_state.check_duration(
                    context.execution_budget,
                    self.definition.name,
                )
                context.execution_budget_state.consume_llm_call(
                    context.execution_budget,
                    self.definition.name,
                )

                context.raise_if_execution_ownership_lost()

                await self._emit(
                    AgentExecutionEvent(
                        event_type=AgentExecutionEventType.LLM_REQUESTED,
                        agent_name=self.definition.name,
                        run_id=context.run_id,
                        session_id=context.session_id,
                        user_id=context.user_id,
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

                context.raise_if_execution_ownership_lost()

                await self._emit(
                    AgentExecutionEvent(
                        event_type=AgentExecutionEventType.LLM_COMPLETED,
                        agent_name=self.definition.name,
                        run_id=context.run_id,
                        session_id=context.session_id,
                        user_id=context.user_id,
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
                    context.raise_if_execution_ownership_lost()

                    await self._emit(
                        AgentExecutionEvent(
                            event_type=AgentExecutionEventType.AGENT_COMPLETED,
                            agent_name=self.definition.name,
                            run_id=context.run_id,
                            session_id=context.session_id,
                            user_id=context.user_id,
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

                context.raise_if_execution_ownership_lost()

                context.execution_budget_state.check_duration(
                    context.execution_budget,
                    self.definition.name,
                )
                context.execution_budget_state.consume_tool_calls(
                    len(result.tool_calls),
                    context.execution_budget,
                    self.definition.name,
                )
                context.execution_budget_state.consume_tool_round(
                    context.execution_budget,
                    self.definition.name,
                )

                tool_rounds += 1

                context.raise_if_execution_ownership_lost()

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
                    run_id=context.run_id,
                    session_id=context.session_id,
                    user_id=context.user_id,
                    tool_round=tool_rounds,
                    metadata={
                        "error_type": type(exc).__name__,
                    },
                )
            )
            raise

    async def run(
        self,
        context: AgentExecutionContext,
    ) -> AgentResponse:
        """
        Execute a new agent interaction.
        """
        if not isinstance(context, AgentExecutionContext):
            raise TypeError("LLMAgent context must be an AgentExecutionContext.")

        context.raise_if_execution_ownership_lost()

        await self._emit(
            AgentExecutionEvent(
                event_type=AgentExecutionEventType.AGENT_STARTED,
                agent_name=self.definition.name,
                run_id=context.run_id,
                session_id=context.session_id,
                user_id=context.user_id,
            )
        )

        orchestration_step_index = self._start_orchestration_step(context)

        try:
            messages = list(context.build_llm_messages())
        except Exception as exc:
            await self._emit(
                AgentExecutionEvent(
                    event_type=AgentExecutionEventType.AGENT_FAILED,
                    agent_name=self.definition.name,
                    run_id=context.run_id,
                    session_id=context.session_id,
                    user_id=context.user_id,
                    tool_round=0,
                    metadata={
                        "error_type": type(exc).__name__,
                    },
                )
            )
            raise

        response = await self._continue(
            context,
            messages,
            tool_rounds=0,
        )

        self._complete_orchestration_step(
            context,
            orchestration_step_index,
            tool_round=response.metadata.get("tool_rounds"),
        )

        return response

    async def resume(
        self,
        context: AgentExecutionContext,
        checkpoint: AgentExecutionCheckpoint,
    ) -> AgentResponse:
        """
        Resume an agent from a durable execution checkpoint.

        The checkpoint conversation is the authoritative continuation
        state. Completed tools represented by the checkpoint are not
        executed again.
        """
        if not isinstance(context, AgentExecutionContext):
            raise TypeError("LLMAgent context must be an AgentExecutionContext.")

        context.raise_if_execution_ownership_lost()

        if not isinstance(
            checkpoint,
            AgentExecutionCheckpoint,
        ):
            raise TypeError("LLMAgent checkpoint must be an AgentExecutionCheckpoint.")

        if checkpoint.agent_name != self.definition.name:
            raise ValueError("Checkpoint agent_name does not match the LLMAgent definition.")

        if context.run_id != checkpoint.run_id:
            raise ValueError("Checkpoint run_id does not match the AgentExecutionContext.")

        if checkpoint.position not in {
            AgentCheckpointPosition.BEFORE_TOOL_EXECUTION,
            AgentCheckpointPosition.AFTER_TOOL_EXECUTION,
        }:
            raise ValueError(
                "LLMAgent can only resume from a before- or after-tool-execution checkpoint."
            )

        context.execution_budget_state = ExecutionBudgetState.from_dict(
            checkpoint.execution_budget_state.to_dict()
        )

        orchestration_step_index = self._start_orchestration_step(context)

        messages = list(checkpoint.messages)
        tool_rounds = checkpoint.tool_round

        if checkpoint.position is AgentCheckpointPosition.BEFORE_TOOL_EXECUTION:
            from ai_platform.agents.llm_messages import (
                assistant_tool_calls_from_message,
            )

            if not messages:
                raise ValueError("Before-tool-execution checkpoint must contain messages.")

            assistant_message = messages[-1]

            tool_calls = assistant_tool_calls_from_message(
                assistant_message,
            )

            context.raise_if_execution_ownership_lost()

            await self._execute_tool_calls_and_append_results(
                messages,
                context,
                tool_calls,
                tool_round=tool_rounds,
            )

        context.raise_if_execution_ownership_lost()

        response = await self._continue(
            context,
            messages,
            tool_rounds=tool_rounds,
        )

        self._complete_orchestration_step(
            context,
            orchestration_step_index,
            tool_round=response.metadata.get("tool_rounds"),
        )

        return response
