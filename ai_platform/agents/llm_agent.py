from __future__ import annotations

from datetime import UTC, datetime

from dataclasses import dataclass

from ai_platform.agents.budget import ExecutionBudgetState
from ai_platform.agents.observer import AgentExecutionObserver
from app.control_plane.agent_run_steps.models import AgentRunStep, AgentRunStepStatus
from ai_platform.agents.checkpoint import (
    AgentCheckpointHandler,
    AgentCheckpointPosition,
    AgentExecutionCheckpoint,
)
from ai_platform.agents.execution import AgentExecutionContext
from ai_platform.agents.exceptions import AgentExecutionOwnershipLostError
from ai_platform.agents.orchestration import (
    OrchestrationStep,
    OrchestrationStepCompletionPolicy,
    OrchestrationStepStatus,
)
from ai_platform.agents.llm_messages import AgentMessage
from ai_platform.agents.models import (
    AgentDefinition,
    AgentResponse,
)
from tools.models import ToolExecutionFailureCategory
from ai_platform.agents.observability import (
    AgentExecutionEvent,
    AgentExecutionEventType,
)


@dataclass(frozen=True)
class _AgentContinuationResult:
    """
    Internal continuation state returned by the LLM/tool execution engine.

    The caller retains the authoritative mutable conversation state so
    orchestration can advance logical steps without rebuilding context.
    """

    response: AgentResponse | None
    tool_rounds: int
    boundary_reached: bool


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

    @classmethod
    def _tool_execution_metadata(
        cls,
        tool_name: str,
        output: object,
        execution_metadata: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Merge bounded execution and tool-specific provenance metadata."""
        metadata: dict[str, object] = {}

        if execution_metadata:
            metadata.update(execution_metadata)

        rag_metadata = cls._tool_provenance_metadata(tool_name, output)
        if rag_metadata:
            metadata.update(rag_metadata)

        return metadata

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

    async def _bind_orchestration_tool_execution(
        self,
        context: AgentExecutionContext,
        tool_calls,
    ) -> None:
        """Bind the actual tool call to the current durable tool step."""
        if context.run_id is None:
            return

        if context.orchestration_plan is None:
            return

        step = context.orchestration_state.current_step

        if step is None:
            return

        if step.completion_policy is not OrchestrationStepCompletionPolicy.ON_TOOL_RESULT:
            return

        expected_tool_name = step.metadata.get("completion_tool_name")

        if not isinstance(expected_tool_name, str) or not expected_tool_name:
            return

        matching_calls = [
            tool_call for tool_call in tool_calls if tool_call.name == expected_tool_name
        ]

        if len(matching_calls) != 1:
            raise RuntimeError(
                "durable orchestration tool binding requires exactly one "
                "matching tool call: "
                f"step={step.step_id}, expected_tool={expected_tool_name}, "
                f"matching_calls={len(matching_calls)}"
            )

        repository = context.get_agent_run_steps_repository()

        if repository is None:
            return

        tool_call = matching_calls[0]

        try:
            existing = repository.get(
                context.run_id,
                step.step_id,
            )

            if existing is None:
                raise RuntimeError(
                    "cannot bind tool execution to missing durable "
                    "orchestration step: "
                    f"{context.run_id}/{step.step_id}"
                )

            if existing.status is not AgentRunStepStatus.RUNNING:
                raise RuntimeError(
                    "cannot execute tool for durable orchestration step "
                    "from status: "
                    f"{existing.status.value}"
                )

            repository.bind_execution(
                context.run_id,
                step.step_id,
                tool_name=tool_call.name,
                call_id=tool_call.call_id,
                input=tool_call.arguments,
            )
        finally:
            repository.close()

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

        await self._bind_orchestration_tool_execution(
            context,
            tool_calls,
        )

        context.raise_if_execution_ownership_lost()

        try:
            tool_results = await context.execute_tool_calls(
                tool_calls,
            )
        except AgentExecutionOwnershipLostError as exc:
            if context.orchestration_plan is not None:
                step = context.orchestration_state.current_step

                if (
                    step is not None
                    and step.completion_policy is OrchestrationStepCompletionPolicy.ON_TOOL_RESULT
                ):
                    await self._persist_orchestration_step_ambiguous(
                        context,
                        step,
                        error=str(exc),
                        failure_category=(ToolExecutionFailureCategory.EXECUTION_AMBIGUOUS.value),
                    )

            raise

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
                        metadata=self._tool_execution_metadata(
                            tool_call.name,
                            tool_result.output,
                            tool_result.metadata,
                        ),
                    )
                )

                await self._record_orchestration_tool_result(
                    context,
                    tool_call,
                    tool_result,
                    tool_round=tool_round,
                )
            else:
                if (
                    context.orchestration_plan is not None
                    and context.orchestration_state.current_step is not None
                ):
                    step = context.orchestration_state.current_step

                    if (
                        step.completion_policy is OrchestrationStepCompletionPolicy.ON_TOOL_RESULT
                        and step.metadata.get("completion_tool_name") == tool_call.name
                    ):
                        failure_category = (
                            tool_result.failure_category.value
                            if tool_result.failure_category is not None
                            else None
                        )

                        if (
                            tool_result.failure_category
                            is ToolExecutionFailureCategory.EXECUTION_AMBIGUOUS
                        ):
                            await self._persist_orchestration_step_ambiguous(
                                context,
                                step,
                                error=(tool_result.error or "Tool execution outcome is ambiguous."),
                                failure_category=failure_category,
                            )
                        elif (
                            tool_result.failure_category
                            is not ToolExecutionFailureCategory.EXECUTION_IN_PROGRESS
                        ):
                            await self._persist_orchestration_step_failed(
                                context,
                                step,
                                error=(tool_result.error or "Tool execution failed."),
                                failure_category=failure_category,
                            )

                failure_metadata = dict(tool_result.metadata)

                if tool_result.failure_category is not None:
                    failure_metadata["failure_category"] = tool_result.failure_category.value

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
                        metadata=failure_metadata,
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
                    metadata=self._orchestration_checkpoint_metadata(context),
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
                    metadata=self._orchestration_checkpoint_metadata(context),
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

    def _orchestration_checkpoint_metadata(
        self,
        context: AgentExecutionContext,
    ) -> dict:
        """
        Return checkpoint metadata with the logical orchestration position.

        Canonical messages remain authoritative for conversation and tool
        execution state. This snapshot only persists bounded logical-step
        lifecycle state so recovery can resume the correct orchestration step.
        """
        metadata = dict(context.metadata)

        if context.orchestration_plan is None:
            return metadata

        state = context.orchestration_state

        metadata["orchestration"] = {
            "current_step_index": state.current_step_index,
            "steps": {
                step.step_id: {
                    "status": step.status.value,
                    "tool_round": step.tool_round,
                }
                for step in state.steps
                if step.status is not OrchestrationStepStatus.PENDING
            },
        }

        return metadata

    def _validate_restored_orchestration_state(
        self,
        context: AgentExecutionContext,
        checkpoint_position: AgentCheckpointPosition,
    ) -> None:
        """
        Validate the restored orchestration lifecycle against the checkpoint
        position.

        BEFORE_TOOL_EXECUTION checkpoints retain a RUNNING current step
        because the tool still needs to execute. AFTER_TOOL_EXECUTION
        checkpoints retain a COMPLETED current step because the tool result
        has already been persisted and the logical step boundary was reached.
        """
        state = context.orchestration_state

        if state.current_step_index is None:
            if any(step.status is not OrchestrationStepStatus.COMPLETED for step in state.steps):
                raise ValueError(
                    "Checkpoint orchestration state without a current step "
                    "must have all steps COMPLETED."
                )
            return

        current_index = state.current_step_index
        current_step = state.steps[current_index]

        if checkpoint_position is AgentCheckpointPosition.BEFORE_TOOL_EXECUTION:
            if current_step.status is not OrchestrationStepStatus.RUNNING:
                raise ValueError(
                    "Checkpoint orchestration current step must be RUNNING "
                    "before tool execution."
                )
        elif checkpoint_position is AgentCheckpointPosition.AFTER_TOOL_EXECUTION:
            if current_step.status is not OrchestrationStepStatus.COMPLETED:
                raise ValueError(
                    "Checkpoint orchestration current step must be COMPLETED "
                    "after tool execution."
                )
        else:
            raise ValueError("Unsupported checkpoint position for orchestration recovery.")

        for index, step in enumerate(state.steps):
            if index < current_index:
                if step.status is not OrchestrationStepStatus.COMPLETED:
                    raise ValueError(
                        "Checkpoint orchestration steps before the current "
                        "step must be COMPLETED."
                    )
            elif index > current_index:
                if step.status is not OrchestrationStepStatus.PENDING:
                    raise ValueError(
                        "Checkpoint orchestration steps after the current " "step must be PENDING."
                    )

    def _restore_orchestration_checkpoint_state(
        self,
        context: AgentExecutionContext,
        checkpoint: AgentExecutionCheckpoint,
    ) -> None:
        """
        Restore bounded logical orchestration lifecycle state from checkpoint
        metadata.

        Step results are intentionally not restored from metadata. Successful
        tool results are reconstructed from canonical checkpoint messages,
        while LLM-produced results remain part of the resumed conversation.
        """
        if context.orchestration_plan is None:
            return

        orchestration_metadata = checkpoint.metadata.get("orchestration")

        if orchestration_metadata is None:
            return

        if not isinstance(orchestration_metadata, dict):
            raise ValueError("Checkpoint orchestration metadata must be a dictionary.")

        current_step_index = orchestration_metadata.get("current_step_index")

        if current_step_index is not None:
            if not isinstance(current_step_index, int) or isinstance(
                current_step_index,
                bool,
            ):
                raise ValueError("Checkpoint orchestration current_step_index must be an integer.")

            if current_step_index < 0 or current_step_index >= len(
                context.orchestration_state.steps
            ):
                raise ValueError("Checkpoint orchestration current_step_index is out of range.")

        steps_metadata = orchestration_metadata.get("steps", {})

        if not isinstance(steps_metadata, dict):
            raise ValueError("Checkpoint orchestration steps metadata must be a dictionary.")

        state = context.orchestration_state

        for step_id, step_snapshot in steps_metadata.items():
            if not isinstance(step_id, str) or not step_id.strip():
                raise ValueError("Checkpoint orchestration step IDs must not be empty.")

            if not isinstance(step_snapshot, dict):
                raise ValueError(
                    f"Checkpoint orchestration step {step_id!r} " "metadata must be a dictionary."
                )

            index = state._resolve_step_id(step_id)
            step = state.steps[index]

            status_value = step_snapshot.get("status")
            tool_round = step_snapshot.get("tool_round")

            if status_value not in {
                OrchestrationStepStatus.RUNNING.value,
                OrchestrationStepStatus.COMPLETED.value,
                OrchestrationStepStatus.FAILED.value,
            }:
                raise ValueError(
                    f"Checkpoint orchestration step {step_id!r} " "has an invalid status."
                )

            if tool_round is not None:
                if not isinstance(tool_round, int) or isinstance(tool_round, bool):
                    raise ValueError(
                        f"Checkpoint orchestration step {step_id!r} "
                        "tool_round must be an integer."
                    )

                if tool_round < 0:
                    raise ValueError(
                        f"Checkpoint orchestration step {step_id!r} "
                        "tool_round must not be negative."
                    )

            restored_status = OrchestrationStepStatus(status_value)

            context.orchestration_state.steps[index] = OrchestrationStep(
                step_id=step.step_id,
                step_index=step.step_index,
                name=step.name,
                status=restored_status,
                completion_policy=step.completion_policy,
                tool_round=tool_round,
                metadata=step.metadata,
            )

        state.current_step_index = current_step_index

        self._validate_restored_orchestration_state(
            context,
            checkpoint.position,
        )

    async def _emit_orchestration_step_event(
        self,
        context: AgentExecutionContext,
        event_type: AgentExecutionEventType,
        step_index: int,
        *,
        metadata: dict[str, object] | None = None,
    ) -> None:
        step = context.orchestration_state.steps[step_index]

        await self._emit(
            AgentExecutionEvent(
                event_type=event_type,
                agent_name=self.definition.name,
                run_id=context.run_id,
                session_id=context.session_id,
                user_id=context.user_id,
                step_id=step.step_id,
                step_index=step.step_index,
                step_name=step.name,
                metadata={} if metadata is None else metadata,
            )
        )

    async def _emit_orchestration_step_started(
        self,
        context: AgentExecutionContext,
        step_index: int,
    ) -> None:
        await self._emit_orchestration_step_event(
            context,
            AgentExecutionEventType.ORCHESTRATION_STEP_STARTED,
            step_index,
        )

    async def _emit_orchestration_step_completed(
        self,
        context: AgentExecutionContext,
        step_index: int,
    ) -> None:
        await self._emit_orchestration_step_event(
            context,
            AgentExecutionEventType.ORCHESTRATION_STEP_COMPLETED,
            step_index,
        )

    async def _emit_orchestration_step_failed(
        self,
        context: AgentExecutionContext,
        step_index: int,
        *,
        error_type: str | None = None,
    ) -> None:
        metadata = {}
        if error_type is not None:
            metadata["error_type"] = error_type

        await self._emit_orchestration_step_event(
            context,
            AgentExecutionEventType.ORCHESTRATION_STEP_FAILED,
            step_index,
            metadata=metadata,
        )

    @staticmethod
    def _durable_step_type(step: OrchestrationStep) -> str:
        """Map an orchestration step to its durable ledger type."""
        if step.completion_policy is OrchestrationStepCompletionPolicy.ON_TOOL_RESULT:
            return "tool"

        return "model"

    async def _persist_orchestration_step_planned(
        self,
        context: AgentExecutionContext,
        step: OrchestrationStep,
    ) -> None:
        """Ensure the logical orchestration step exists as PLANNED."""
        if context.run_id is None:
            return

        repository = context.get_agent_run_steps_repository()
        if repository is None:
            return

        try:
            existing = repository.get(
                context.run_id,
                step.step_id,
            )

            if existing is not None:
                return

            repository.create(
                AgentRunStep(
                    run_id=context.run_id,
                    step_id=step.step_id,
                    step_index=step.step_index,
                    step_type=self._durable_step_type(step),
                    status=AgentRunStepStatus.PLANNED,
                    metadata=dict(step.metadata),
                )
            )
        finally:
            repository.close()

    async def _persist_orchestration_step_running(
        self,
        context: AgentExecutionContext,
        step: OrchestrationStep,
    ) -> None:
        """Ensure the durable orchestration step is RUNNING."""
        if context.run_id is None:
            return

        repository = context.get_agent_run_steps_repository()
        if repository is None:
            return

        try:
            existing = repository.get(
                context.run_id,
                step.step_id,
            )

            if existing is None:
                repository.create(
                    AgentRunStep(
                        run_id=context.run_id,
                        step_id=step.step_id,
                        step_index=step.step_index,
                        step_type=self._durable_step_type(step),
                        status=AgentRunStepStatus.PLANNED,
                        metadata=dict(step.metadata),
                    )
                )
                existing = repository.get(
                    context.run_id,
                    step.step_id,
                )

            if existing is None:
                raise RuntimeError(
                    "durable orchestration step disappeared after creation: "
                    f"{context.run_id}/{step.step_id}"
                )

            if existing.status is AgentRunStepStatus.RUNNING:
                return

            if existing.status is not AgentRunStepStatus.PLANNED:
                raise RuntimeError(
                    "cannot start durable orchestration step from status: "
                    f"{existing.status.value}"
                )

            now = datetime.now(UTC)

            repository.transition(
                context.run_id,
                step.step_id,
                status=AgentRunStepStatus.RUNNING,
                updated_at=now,
                started_at=now,
            )
        finally:
            repository.close()

    async def _persist_orchestration_step_completed(
        self,
        context: AgentExecutionContext,
        step: OrchestrationStep,
        *,
        output: object | None = None,
    ) -> None:
        """Persist successful completion of an orchestration step."""
        if context.run_id is None:
            return

        repository = context.get_agent_run_steps_repository()
        if repository is None:
            return

        try:
            existing = repository.get(
                context.run_id,
                step.step_id,
            )

            if existing is None:
                raise RuntimeError(
                    "cannot complete missing durable orchestration step: "
                    f"{context.run_id}/{step.step_id}"
                )

            if existing.status is AgentRunStepStatus.COMPLETED:
                return

            now = datetime.now(UTC)

            step_result = context.orchestration_state.get_step_result(step.step_id)
            metadata = (
                dict(step_result.metadata) if step_result is not None else dict(step.metadata)
            )

            repository.transition(
                context.run_id,
                step.step_id,
                status=AgentRunStepStatus.COMPLETED,
                updated_at=now,
                completed_at=now,
                output=output,
                metadata=metadata,
            )
        finally:
            repository.close()

    async def _persist_orchestration_step_ambiguous(
        self,
        context: AgentExecutionContext,
        step: OrchestrationStep,
        *,
        error: str,
        failure_category: str | None = None,
    ) -> None:
        """Persist an orchestration step whose external outcome is uncertain."""
        if context.run_id is None:
            return

        repository = context.get_agent_run_steps_repository()
        if repository is None:
            return

        try:
            existing = repository.get(
                context.run_id,
                step.step_id,
            )

            if existing is None:
                raise RuntimeError(
                    "cannot mark missing durable orchestration step ambiguous: "
                    f"{context.run_id}/{step.step_id}"
                )

            if existing.status is AgentRunStepStatus.AMBIGUOUS:
                return

            if existing.status is AgentRunStepStatus.COMPLETED:
                return

            now = datetime.now(UTC)

            repository.transition(
                context.run_id,
                step.step_id,
                status=AgentRunStepStatus.AMBIGUOUS,
                updated_at=now,
                completed_at=now,
                error=error,
                failure_category=failure_category,
            )
        finally:
            repository.close()

    async def _persist_orchestration_step_failed(
        self,
        context: AgentExecutionContext,
        step: OrchestrationStep,
        *,
        error: str,
        failure_category: str | None = None,
    ) -> None:
        """Persist failed execution of an orchestration step."""
        if context.run_id is None:
            return

        repository = context.get_agent_run_steps_repository()
        if repository is None:
            return

        try:
            existing = repository.get(
                context.run_id,
                step.step_id,
            )

            if existing is None:
                raise RuntimeError(
                    "cannot fail missing durable orchestration step: "
                    f"{context.run_id}/{step.step_id}"
                )

            if existing.status in {
                AgentRunStepStatus.FAILED,
                AgentRunStepStatus.AMBIGUOUS,
            }:
                return

            now = datetime.now(UTC)

            repository.transition(
                context.run_id,
                step.step_id,
                status=AgentRunStepStatus.FAILED,
                updated_at=now,
                completed_at=now,
                error=error,
                failure_category=failure_category,
            )
        finally:
            repository.close()

    async def _start_orchestration_step(
        self,
        context: AgentExecutionContext,
    ) -> int | None:
        """Start the current orchestration step and persist its lifecycle."""
        state = context.orchestration_state

        if context.orchestration_plan is None:
            return None

        step = state.current_step

        if step is None:
            step = state.steps[0]

            await self._persist_orchestration_step_planned(
                context,
                step,
            )

            state.start_step(0)

        elif step.status is OrchestrationStepStatus.PENDING:
            await self._persist_orchestration_step_planned(
                context,
                step,
            )

            state.start_step(step.step_index)

        await self._persist_orchestration_step_running(
            context,
            step,
        )

        await self._emit_orchestration_step_started(
            context,
            step.step_index,
        )

        return step.step_index

    def _set_orchestration_step_result(
        self,
        context: AgentExecutionContext,
        step_index: int | None,
        response: AgentResponse,
    ) -> None:
        """Publish the completed interaction as the logical step result."""
        if step_index is None:
            return

        step = context.orchestration_state.steps[step_index]

        if step.completion_policy is not OrchestrationStepCompletionPolicy.ON_AGENT_RESPONSE:
            return

        context.orchestration_state.set_step_result(
            step.step_id,
            response.output,
            metadata=dict(response.metadata),
        )

    async def _complete_orchestration_step(
        self,
        context: AgentExecutionContext,
        step_index: int | None,
        *,
        tool_round: int | None = None,
    ) -> None:
        if step_index is None:
            return

        state = context.orchestration_state
        step = state.steps[step_index]

        if step.completion_policy is not OrchestrationStepCompletionPolicy.ON_AGENT_RESPONSE:
            return

        state.complete_step(
            step_index,
            tool_round=tool_round,
        )

        completed_step = state.steps[step_index]
        step_result = state.get_step_result(completed_step.step_id)

        await self._persist_orchestration_step_completed(
            context,
            completed_step,
            output=step_result.output if step_result is not None else None,
        )

        await self._emit_orchestration_step_completed(
            context,
            step_index,
        )

    async def _record_orchestration_tool_result(
        self,
        context: AgentExecutionContext,
        tool_call,
        tool_result,
        *,
        tool_round: int,
    ) -> None:
        """
        Complete a logical orchestration step at its configured tool boundary.

        Tool rounds remain execution mechanics. A logical step is completed
        only when its declared completion policy and expected tool match the
        successful tool result.
        """
        if context.orchestration_plan is None:
            return

        state = context.orchestration_state
        step = state.current_step

        if step is None:
            return

        if step.completion_policy is not OrchestrationStepCompletionPolicy.ON_TOOL_RESULT:
            return

        expected_tool_name = step.metadata.get("completion_tool_name")

        if expected_tool_name != tool_call.name:
            return

        if not tool_result.success:
            return

        metadata = self._tool_execution_metadata(
            tool_call.name,
            tool_result.output,
            tool_result.metadata,
        )

        state.set_step_result(
            step.step_id,
            tool_result.output,
            metadata=metadata,
        )

        state.complete_step(
            step.step_index,
            tool_round=tool_round,
        )

        completed_step = state.steps[step.step_index]

        await self._persist_orchestration_step_completed(
            context,
            completed_step,
            output=tool_result.output,
        )

        await self._emit_orchestration_step_completed(
            context,
            step.step_index,
        )

    def _restore_orchestration_tool_result_from_messages(
        self,
        context: AgentExecutionContext,
        messages,
        *,
        tool_round: int,
    ) -> None:
        """
        Restore a completed tool-bound orchestration step from checkpoint messages.

        This path is used only after tool execution has already happened.
        It reconstructs the logical step result without executing the tool again
        and is idempotent when checkpoint metadata already marks the step
        completed.
        """
        if context.orchestration_plan is None:
            return

        state = context.orchestration_state
        step = state.current_step

        if step is None:
            return

        if step.completion_policy is not OrchestrationStepCompletionPolicy.ON_TOOL_RESULT:
            return

        expected_tool_name = step.metadata.get("completion_tool_name")

        if not isinstance(expected_tool_name, str) or not expected_tool_name.strip():
            return

        import json

        for message in reversed(messages):
            if getattr(message.role, "value", None) != "tool":
                continue

            try:
                payload = json.loads(message.content)
            except (TypeError, json.JSONDecodeError):
                continue

            if not isinstance(payload, dict):
                continue

            if payload.get("tool_name") != expected_tool_name:
                continue

            if payload.get("success") is not True:
                continue

            if "output" not in payload:
                continue

            output = payload["output"]

            metadata = self._tool_provenance_metadata(
                expected_tool_name,
                output,
            )

            state.set_step_result(
                step.step_id,
                output,
                metadata=metadata,
            )

            if step.status is OrchestrationStepStatus.COMPLETED:
                return

            if step.status is not OrchestrationStepStatus.RUNNING:
                raise RuntimeError(
                    "Cannot restore orchestration tool result for a step "
                    f"in {step.status.value!r} status."
                )

            state.complete_step(
                step.step_index,
                tool_round=tool_round,
            )
            return

    def _advance_orchestration_step(
        self,
        context: AgentExecutionContext,
    ) -> int | None:
        """Advance to the next logical orchestration step, if configured."""
        if context.orchestration_plan is None:
            return None

        next_step = context.orchestration_state.advance()

        if next_step is None:
            return None

        return next_step.step_index

    async def _continue(
        self,
        context: AgentExecutionContext,
        messages: list[AgentMessage],
        *,
        tool_rounds: int,
    ) -> _AgentContinuationResult:
        """
        Continue an agent execution from an existing conversation state.

        Control returns to the orchestration coordinator when the current
        logical step reaches its configured completion boundary.

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

                    if context.orchestration_plan is None:
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

                    response = AgentResponse(
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

                    boundary_reached = context.orchestration_plan is None

                    if context.orchestration_plan is not None:
                        current_step = context.orchestration_state.current_step
                        boundary_reached = (
                            current_step is not None
                            and current_step.completion_policy
                            is OrchestrationStepCompletionPolicy.ON_AGENT_RESPONSE
                        )

                    return _AgentContinuationResult(
                        response=response,
                        tool_rounds=tool_rounds,
                        boundary_reached=boundary_reached,
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

                if (
                    context.orchestration_plan is not None
                    and context.orchestration_state.current_step is not None
                    and context.orchestration_state.current_step.status
                    is OrchestrationStepStatus.COMPLETED
                ):
                    return _AgentContinuationResult(
                        response=None,
                        tool_rounds=tool_rounds,
                        boundary_reached=True,
                    )

        except Exception as exc:
            if context.orchestration_plan is not None:
                current_step = context.orchestration_state.current_step

                if (
                    current_step is not None
                    and current_step.status is OrchestrationStepStatus.RUNNING
                ):
                    failed_step_index = current_step.step_index
                    context.orchestration_state.fail_step(
                        failed_step_index,
                        metadata={
                            "error_type": type(exc).__name__,
                        },
                    )

                    await self._emit_orchestration_step_failed(
                        context,
                        failed_step_index,
                        error_type=type(exc).__name__,
                    )

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

        When an orchestration plan is configured, the logical steps are
        coordinated around the single LLM/tool continuation engine. The
        conversation messages remain authoritative across step boundaries.
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

        orchestration_step_index = await self._start_orchestration_step(context)

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

        if context.orchestration_plan is None:
            continuation = await self._continue(
                context,
                messages,
                tool_rounds=0,
            )
            if continuation.response is None:
                raise RuntimeError("LLM continuation reached a boundary without a response.")
            return continuation.response

        while True:
            context.raise_if_execution_ownership_lost()

            continuation = await self._continue(
                context,
                messages,
                tool_rounds=(
                    0
                    if orchestration_step_index is None
                    else context.orchestration_state.steps[orchestration_step_index].tool_round or 0
                ),
            )

            if continuation.response is not None:
                self._set_orchestration_step_result(
                    context,
                    orchestration_step_index,
                    continuation.response,
                )

                await self._complete_orchestration_step(
                    context,
                    orchestration_step_index,
                    tool_round=continuation.tool_rounds,
                )

            if not continuation.boundary_reached:
                raise RuntimeError(
                    "Orchestration continuation returned without a completion boundary."
                )

            current_step = context.orchestration_state.current_step
            if current_step is None:
                return continuation.response  # pragma: no cover

            if current_step.status is not OrchestrationStepStatus.COMPLETED:
                raise RuntimeError(
                    "Orchestration boundary was reached before the current " "step completed."
                )

            next_step_index = self._advance_orchestration_step(context)

            if next_step_index is None:
                if continuation.response is None:
                    raise RuntimeError("Final orchestration step completed without a response.")

                await self._emit(
                    AgentExecutionEvent(
                        event_type=AgentExecutionEventType.AGENT_COMPLETED,
                        agent_name=self.definition.name,
                        run_id=context.run_id,
                        session_id=context.session_id,
                        user_id=context.user_id,
                        tool_round=continuation.tool_rounds,
                        provider=continuation.response.metadata.get("provider"),
                        model=continuation.response.metadata.get("model"),
                    )
                )

                return continuation.response

            orchestration_step_index = await self._start_orchestration_step(context)

    async def resume(
        self,
        context: AgentExecutionContext,
        checkpoint: AgentExecutionCheckpoint,
    ) -> AgentResponse:
        """
        Resume an agent from a durable execution checkpoint.

        The checkpoint conversation is the authoritative continuation state.
        Logical orchestration position is restored from bounded checkpoint
        metadata, while completed tools represented by the checkpoint are not
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

        self._restore_orchestration_checkpoint_state(
            context,
            checkpoint,
        )

        messages = list(checkpoint.messages)
        tool_rounds = checkpoint.tool_round

        # Non-orchestrated agents retain the existing recovery semantics.
        if context.orchestration_plan is None:
            if checkpoint.position is AgentCheckpointPosition.BEFORE_TOOL_EXECUTION:
                from ai_platform.agents.llm_messages import (
                    assistant_tool_calls_from_message,
                )

                if not messages:
                    raise ValueError("Before-tool-execution checkpoint must contain messages.")

                tool_calls = assistant_tool_calls_from_message(messages[-1])

                context.raise_if_execution_ownership_lost()

                await self._execute_tool_calls_and_append_results(
                    messages,
                    context,
                    tool_calls,
                    tool_round=tool_rounds,
                )
            else:
                self._restore_orchestration_tool_result_from_messages(
                    context,
                    messages,
                    tool_round=tool_rounds,
                )

            context.raise_if_execution_ownership_lost()

            continuation = await self._continue(
                context,
                messages,
                tool_rounds=tool_rounds,
            )

            response = continuation.response

            if response is None:
                raise RuntimeError("LLM continuation did not produce a response during resume.")

            return response

        state = context.orchestration_state

        # A checkpoint without an orchestration snapshot represents a
        # pre-orchestration checkpoint. Start from the first logical step.
        if state.current_step is None:
            orchestration_step_index = await self._start_orchestration_step(context)
        else:
            orchestration_step_index = state.current_step.step_index

        if orchestration_step_index is None:
            raise RuntimeError("Orchestration recovery could not determine the current step.")

        current_step = state.current_step

        if current_step is None:
            raise RuntimeError("Orchestration recovery has no current step.")

        # A BEFORE_TOOL checkpoint represents an in-progress tool-bound
        # logical step. Execute the captured tool call exactly once.
        if checkpoint.position is AgentCheckpointPosition.BEFORE_TOOL_EXECUTION:
            from ai_platform.agents.llm_messages import (
                assistant_tool_calls_from_message,
            )

            if not messages:
                raise ValueError("Before-tool-execution checkpoint must contain messages.")

            tool_calls = assistant_tool_calls_from_message(messages[-1])

            context.raise_if_execution_ownership_lost()

            await self._execute_tool_calls_and_append_results(
                messages,
                context,
                tool_calls,
                tool_round=tool_rounds,
            )

        else:
            # An AFTER_TOOL checkpoint already contains the successful tool
            # result. Reconstruct the logical tool-bound step without replaying
            # the side effect.
            self._restore_orchestration_tool_result_from_messages(
                context,
                messages,
                tool_round=tool_rounds,
            )

        context.raise_if_execution_ownership_lost()

        # The current recovered step is now complete if it was tool-bound.
        current_step = state.current_step

        if (
            current_step is not None
            and current_step.completion_policy is OrchestrationStepCompletionPolicy.ON_TOOL_RESULT
            and current_step.status is OrchestrationStepStatus.COMPLETED
        ):
            next_step_index = self._advance_orchestration_step(context)

            if next_step_index is None:
                raise RuntimeError(
                    "Orchestration recovery completed without a final response step."
                )

            orchestration_step_index = await self._start_orchestration_step(context)

        elif current_step is not None and current_step.status is OrchestrationStepStatus.COMPLETED:
            next_step_index = self._advance_orchestration_step(context)

            if next_step_index is None:
                raise RuntimeError("Orchestration recovery completed without a final response.")

            orchestration_step_index = next_step_index

        while True:
            context.raise_if_execution_ownership_lost()

            current_step = state.current_step

            if current_step is None:
                raise RuntimeError("Orchestration recovery has no current step.")

            if current_step.status is not OrchestrationStepStatus.RUNNING:
                raise RuntimeError(
                    "Orchestration recovery current step must be RUNNING " "before continuation."
                )

            continuation = await self._continue(
                context,
                messages,
                tool_rounds=(
                    current_step.tool_round if current_step.tool_round is not None else tool_rounds
                ),
            )

            if not continuation.boundary_reached:
                raise RuntimeError(
                    "LLM continuation did not reach an orchestration boundary " "during resume."
                )

            response = continuation.response

            if response is not None:
                self._set_orchestration_step_result(
                    context,
                    orchestration_step_index,
                    response,
                )

            await self._complete_orchestration_step(
                context,
                orchestration_step_index,
                tool_round=continuation.tool_rounds,
            )

            current_step = state.current_step

            if current_step is None:
                raise RuntimeError("Orchestration recovery lost the current step after completion.")

            next_step_index = self._advance_orchestration_step(context)

            if next_step_index is None:
                if response is None:
                    raise RuntimeError(
                        "Final orchestration recovery step completed without " "an agent response."
                    )

                return response

            orchestration_step_index = await self._start_orchestration_step(context)
            tool_rounds = continuation.tool_rounds
