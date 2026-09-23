from __future__ import annotations

from datetime import datetime
from threading import Lock

from app.control_plane.agent_run_steps.exceptions import (
    DuplicateAgentRunStepError,
)
from app.control_plane.agent_run_steps.models import (
    AgentRunStep,
    AgentRunStepStatus,
)


class InMemoryAgentRunStepsRepository:
    def close(self) -> None:
        """No-op close for the in-memory repository."""
        return None

    def __init__(self) -> None:
        self._steps: dict[tuple[str, str], AgentRunStep] = {}
        self._lock = Lock()

    def create(
        self,
        step: AgentRunStep,
        *,
        commit: bool = True,
    ) -> AgentRunStep:
        del commit

        key = (step.run_id, step.step_id)

        with self._lock:
            if key in self._steps:
                raise DuplicateAgentRunStepError(
                    f"agent run step already exists: {step.run_id}/{step.step_id}"
                )

            self._steps[key] = step

        return step

    def get(
        self,
        run_id: str,
        step_id: str,
    ) -> AgentRunStep | None:
        with self._lock:
            return self._steps.get((run_id, step_id))

    def list(
        self,
        run_id: str,
        *,
        status: AgentRunStepStatus | None = None,
        limit: int = 100,
    ) -> list[AgentRunStep]:
        if limit <= 0:
            raise ValueError("limit must be greater than zero.")

        with self._lock:
            steps = [
                step
                for (stored_run_id, _), step in self._steps.items()
                if stored_run_id == run_id and (status is None or step.status == status)
            ]

        return sorted(
            steps,
            key=lambda step: (step.step_index, step.step_id),
        )[:limit]

    def transition(
        self,
        run_id: str,
        step_id: str,
        *,
        status: AgentRunStepStatus,
        updated_at: datetime,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
        output=None,
        error: str | None = None,
        failure_category: str | None = None,
        commit: bool = True,
    ) -> AgentRunStep | None:
        del commit

        key = (run_id, step_id)

        with self._lock:
            step = self._steps.get(key)

            if step is None:
                return None

            updated = step.transition_to(status)

            if status == AgentRunStepStatus.RUNNING:
                completed_at_value = None
            else:
                if completed_at is None:
                    raise ValueError(
                        "completed_at is required for terminal agent run step "
                        f"status: {status.value}"
                    )
                completed_at_value = completed_at

            updated = updated.model_copy(
                update={
                    "updated_at": updated_at,
                    "started_at": (started_at if started_at is not None else step.started_at),
                    "completed_at": completed_at_value,
                    "output": output if output is not None else step.output,
                    "error": error,
                    "failure_category": failure_category,
                }
            )

            self._steps[key] = updated
            return updated

    def bind_execution(
        self,
        run_id: str,
        step_id: str,
        *,
        tool_name: str,
        call_id: str,
        input: object | None = None,
        commit: bool = True,
    ) -> AgentRunStep | None:
        del commit

        key = (run_id, step_id)

        with self._lock:
            step = self._steps.get(key)

            if step is None:
                return None

            if step.status is not AgentRunStepStatus.RUNNING:
                raise ValueError(
                    "agent run step execution binding requires RUNNING status: "
                    f"{step.status.value}"
                )

            if step.tool_name is not None and step.tool_name != tool_name:
                raise ValueError(
                    "agent run step tool_name is already bound to a different "
                    f"value: {step.tool_name}"
                )

            if step.call_id is not None and step.call_id != call_id:
                raise ValueError(
                    "agent run step call_id is already bound to a different "
                    f"value: {step.call_id}"
                )

            if step.input is not None and step.input != input:
                raise ValueError(
                    "agent run step input is already bound to a different " f"value: {step.input}"
                )

            updated = step.model_copy(
                update={
                    "tool_name": tool_name,
                    "call_id": call_id,
                    "input": input,
                }
            )

            self._steps[key] = updated
            return updated

    def retry(
        self,
        run_id: str,
        step_id: str,
        *,
        updated_at: datetime,
        started_at: datetime | None = None,
        commit: bool = True,
    ) -> AgentRunStep | None:
        del commit

        key = (run_id, step_id)

        with self._lock:
            step = self._steps.get(key)

            if step is None:
                return None

            if step.status != AgentRunStepStatus.FAILED:
                raise ValueError(
                    "agent run step retry requires FAILED status: " f"{step.status.value}"
                )

            updated = step.model_copy(
                update={
                    "status": AgentRunStepStatus.RUNNING,
                    "attempt": step.attempt + 1,
                    "started_at": started_at,
                    "completed_at": None,
                    "output": None,
                    "error": None,
                    "failure_category": None,
                    "updated_at": updated_at,
                }
            )

            self._steps[key] = updated
            return updated

    def clear(self) -> None:
        with self._lock:
            self._steps.clear()
