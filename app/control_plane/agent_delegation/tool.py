from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from dataclasses import replace
from typing import Any, Callable

from tools.execution.context import ToolExecutionContext
from tools.models import ToolDefinition, ToolProvider

from app.control_plane.agent_delegation.models import (
    AgentDelegationRequest,
    AgentDelegationResult,
)
from app.control_plane.agent_delegation.service import AgentDelegationService
from app.control_plane.agent_runs.models import AgentRunStatus


class AgentDelegationTool:
    """
    Control-plane tool for synchronous delegation to another agent.

    The LLM supplies only the delegation target, task input, optional
    idempotency key, and bounded metadata. Parent identity, hierarchy,
    governance, and execution policy remain control-plane owned.
    """

    def __init__(
        self,
        service_scope_factory: Callable[[], AbstractAsyncContextManager[AgentDelegationService]],
    ) -> None:
        self._service_scope_factory = service_scope_factory

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="agent.delegate",
            description=(
                "Delegate a task to another enabled enterprise agent and "
                "return the child agent's durable execution result."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "agent_name": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Name of the enabled child agent.",
                    },
                    "input": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Task to delegate to the child agent.",
                    },
                    "idempotency_key": {
                        "type": "string",
                        "minLength": 1,
                        "description": (
                            "Optional stable key for safely retrying the same " "delegation."
                        ),
                    },
                    "metadata": {
                        "type": "object",
                        "additionalProperties": True,
                        "description": "Optional delegation metadata.",
                    },
                },
                "required": ["agent_name", "input"],
                "additionalProperties": False,
            },
            provider=ToolProvider(
                kind="native",
                name="control-plane",
            ),
            metadata={
                "category": "orchestration",
                "read_only": False,
            },
        )

    async def execute(
        self,
        arguments: dict[str, Any],
    ) -> Any:
        raise RuntimeError(
            "agent.delegate requires an execution context.",
        )

    async def execute_with_context(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> Any:
        parent_run_id = context.run_id

        if parent_run_id is None or not parent_run_id.strip():
            raise ValueError(
                "agent.delegate requires a durable parent run_id.",
            )

        agent_name = arguments.get("agent_name")
        child_input = arguments.get("input")

        if not isinstance(agent_name, str) or not agent_name.strip():
            raise ValueError(
                "agent.delegate agent_name must be a non-empty string.",
            )

        if not isinstance(child_input, str) or not child_input.strip():
            raise ValueError(
                "agent.delegate input must be a non-empty string.",
            )

        idempotency_key = arguments.get("idempotency_key")

        if idempotency_key is not None:
            if not isinstance(idempotency_key, str) or not idempotency_key.strip():
                raise ValueError(
                    "agent.delegate idempotency_key must be a non-empty string.",
                )

        metadata = arguments.get("metadata", {})

        if not isinstance(metadata, dict):
            raise TypeError(
                "agent.delegate metadata must be an object.",
            )

        scope = self._service_scope_factory()

        async with scope as service:
            parent_run = service.get_parent_run(parent_run_id)

            if parent_run is None:
                raise ValueError(
                    f"Parent agent run '{parent_run_id}' does not exist.",
                )

            if parent_run.status not in {
                AgentRunStatus.RUNNING,
                AgentRunStatus.WAITING_FOR_APPROVAL,
            }:
                raise ValueError(
                    f"Parent agent run '{parent_run_id}' is not executable in "
                    f"status '{parent_run.status.value}'.",
                )

            self._validate_parent_identity(parent_run, context)

            snapshot = parent_run.request_snapshot

            if snapshot is None:
                raise ValueError(
                    f"Parent agent run '{parent_run_id}' has no request snapshot.",
                )

            parent_request = snapshot.to_request(
                session_id=context.session_id,
                user_id=context.user_id,
                principal=context.principal,
            )

            parent_request = replace(
                parent_request,
                tenant_id=context.tenant_id,
            )

            child_request = replace(
                parent_request,
                input=child_input.strip(),
                metadata={
                    **parent_request.metadata,
                    **dict(metadata),
                },
            )

            result: AgentDelegationResult = await service.execute_delegation(
                AgentDelegationRequest(
                    parent_run_id=parent_run_id,
                    child_agent_name=agent_name.strip(),
                    child_request=child_request,
                    idempotency_key=(
                        idempotency_key.strip() if isinstance(idempotency_key, str) else None
                    ),
                    metadata=dict(metadata),
                )
            )

            child_run = result.child_run

            return {
                "child_run_id": child_run.run_id,
                "agent_name": child_run.agent_name,
                "status": child_run.status.value,
                "output": child_run.output,
                "parent_step_id": result.parent_step_id,
                "created": result.created,
            }

    @staticmethod
    def _validate_parent_identity(
        parent_run: Any,
        context: ToolExecutionContext,
    ) -> None:
        identity_fields = (
            ("session_id", parent_run.session_id, context.session_id),
            ("user_id", parent_run.user_id, context.user_id),
            ("principal", parent_run.principal, context.principal),
            ("tenant_id", parent_run.tenant_id, context.tenant_id),
        )

        for field_name, persisted_value, context_value in identity_fields:
            if persisted_value != context_value:
                raise ValueError(
                    f"agent.delegate parent {field_name} does not match "
                    "the trusted execution context.",
                )
