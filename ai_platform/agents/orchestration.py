from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class OrchestrationStepCompletionPolicy(StrEnum):
    ON_AGENT_RESPONSE = "on_agent_response"


class OrchestrationStepStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class OrchestrationStep:
    """
    Provider-neutral description of one logical agent orchestration step.

    An orchestration step is intentionally distinct from a tool round:
    tool rounds describe LLM/tool interaction budget, while orchestration
    steps describe logical execution progression.
    """

    step_id: str
    step_index: int
    name: str
    status: OrchestrationStepStatus
    completion_policy: OrchestrationStepCompletionPolicy = (
        OrchestrationStepCompletionPolicy.ON_AGENT_RESPONSE
    )
    tool_round: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.step_id, str) or not self.step_id.strip():
            raise ValueError("Orchestration step_id must not be empty.")

        if not isinstance(self.step_index, int) or isinstance(self.step_index, bool):
            raise TypeError("Orchestration step_index must be an integer.")

        if self.step_index < 0:
            raise ValueError("Orchestration step_index must not be negative.")

        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Orchestration step name must not be empty.")

        if not isinstance(self.status, OrchestrationStepStatus):
            raise TypeError("Orchestration step status must be an OrchestrationStepStatus.")

        if not isinstance(
            self.completion_policy,
            OrchestrationStepCompletionPolicy,
        ):
            raise TypeError(
                "Orchestration step completion_policy must be an "
                "OrchestrationStepCompletionPolicy."
            )

        if self.tool_round is not None:
            if not isinstance(self.tool_round, int) or isinstance(self.tool_round, bool):
                raise TypeError("Orchestration step tool_round must be an integer.")

            if self.tool_round < 0:
                raise ValueError("Orchestration step tool_round must not be negative.")

        if not isinstance(self.metadata, dict):
            raise TypeError("Orchestration step metadata must be a dictionary.")

        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class OrchestrationPlan:
    """
    Immutable definition of the logical steps an agent execution should run.

    A plan contains only initial PENDING steps. Runtime transitions belong to
    OrchestrationState.
    """

    steps: tuple[OrchestrationStep, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.steps, tuple):
            object.__setattr__(self, "steps", tuple(self.steps))

        if not self.steps:
            raise ValueError("Orchestration plan must contain at least one step.")

        step_ids: set[str] = set()

        for expected_index, step in enumerate(self.steps):
            if not isinstance(step, OrchestrationStep):
                raise TypeError(
                    "Orchestration plan steps must contain OrchestrationStep instances."
                )

            if step.step_id in step_ids:
                raise ValueError(f"Orchestration plan contains duplicate step_id {step.step_id!r}.")

            if step.step_index != expected_index:
                raise ValueError(
                    "Orchestration plan step indexes must be contiguous " "and begin at zero."
                )

            if step.status is not OrchestrationStepStatus.PENDING:
                raise ValueError(
                    f"Orchestration plan step {step.step_id!r} " "must start in PENDING status."
                )

            if step.tool_round is not None:
                raise ValueError(
                    f"Orchestration plan step {step.step_id!r} "
                    "must not have a runtime tool_round."
                )

            step_ids.add(step.step_id)

    def materialize_state(self) -> OrchestrationState:
        """
        Create a fresh mutable execution state from this immutable plan.
        """
        steps = tuple(
            OrchestrationStep(
                step_id=step.step_id,
                step_index=step.step_index,
                name=step.name,
                status=OrchestrationStepStatus.PENDING,
                completion_policy=step.completion_policy,
                metadata=step.metadata,
            )
            for step in self.steps
        )

        return OrchestrationState(steps=list(steps))


@dataclass
class OrchestrationState:
    """
    Mutable execution state for logical orchestration progression.
    """

    steps: list[OrchestrationStep] = field(default_factory=list)
    current_step_index: int | None = None

    def __post_init__(self) -> None:
        if self.current_step_index is not None:
            if not isinstance(self.current_step_index, int) or isinstance(
                self.current_step_index, bool
            ):
                raise TypeError("Orchestration current_step_index must be an integer.")

            if self.current_step_index < 0:
                raise ValueError("Orchestration current_step_index must not be negative.")

        self.steps = list(self.steps)

        for step in self.steps:
            if not isinstance(step, OrchestrationStep):
                raise TypeError("Orchestration steps must contain OrchestrationStep instances.")

        if self.current_step_index is not None and self.current_step_index >= len(self.steps):
            raise ValueError("Orchestration current_step_index must reference an existing step.")

    @property
    def current_step(self) -> OrchestrationStep | None:
        """Return the current logical step, if one is selected."""
        if self.current_step_index is None:
            return None

        return self.steps[self.current_step_index]

    def start_step(self, step_index: int) -> OrchestrationStep:
        """
        Select a step and transition it to RUNNING.

        The immutable step record is replaced with an updated record so
        callers cannot mutate a step behind the state's back.
        """
        self._validate_step_index(step_index)

        step = self.steps[step_index]

        if step.status not in {
            OrchestrationStepStatus.PENDING,
            OrchestrationStepStatus.RUNNING,
        }:
            raise ValueError(
                f"Orchestration step {step.step_id!r} cannot start from "
                f"status {step.status.value!r}."
            )

        updated = OrchestrationStep(
            step_id=step.step_id,
            step_index=step.step_index,
            name=step.name,
            status=OrchestrationStepStatus.RUNNING,
            completion_policy=step.completion_policy,
            tool_round=step.tool_round,
            metadata=step.metadata,
        )

        self.steps[step_index] = updated
        self.current_step_index = step_index

        return updated

    def complete_step(
        self,
        step_index: int | None = None,
        *,
        tool_round: int | None = None,
    ) -> OrchestrationStep:
        """Transition the selected step to COMPLETED."""
        index = self._resolve_step_index(step_index)
        step = self.steps[index]

        if step.status is not OrchestrationStepStatus.RUNNING:
            raise ValueError(
                f"Orchestration step {step.step_id!r} must be RUNNING before completion."
            )

        if tool_round is not None:
            if not isinstance(tool_round, int) or isinstance(tool_round, bool):
                raise TypeError("Orchestration tool_round must be an integer.")

            if tool_round < 0:
                raise ValueError("Orchestration tool_round must not be negative.")

        updated = OrchestrationStep(
            step_id=step.step_id,
            step_index=step.step_index,
            name=step.name,
            status=OrchestrationStepStatus.COMPLETED,
            completion_policy=step.completion_policy,
            tool_round=tool_round if tool_round is not None else step.tool_round,
            metadata=step.metadata,
        )

        self.steps[index] = updated
        return updated

    def fail_step(
        self,
        step_index: int | None = None,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> OrchestrationStep:
        """Transition the selected step to FAILED."""
        index = self._resolve_step_index(step_index)
        step = self.steps[index]

        if step.status is not OrchestrationStepStatus.RUNNING:
            raise ValueError(f"Orchestration step {step.step_id!r} must be RUNNING before failure.")

        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError("Orchestration failure metadata must be a dictionary.")

        updated_metadata = (
            dict(step.metadata) if metadata is None else {**step.metadata, **metadata}
        )

        updated = OrchestrationStep(
            step_id=step.step_id,
            step_index=step.step_index,
            name=step.name,
            status=OrchestrationStepStatus.FAILED,
            completion_policy=step.completion_policy,
            tool_round=step.tool_round,
            metadata=updated_metadata,
        )

        self.steps[index] = updated
        return updated

    def advance(self) -> OrchestrationStep | None:
        """
        Move to the next logical step.

        The current step must already be COMPLETED. Returns None when
        there is no next step.
        """
        if self.current_step_index is None:
            raise ValueError("Cannot advance orchestration without a current step.")

        current = self.current_step

        if current is None:
            raise ValueError("Orchestration current step is invalid.")

        if current.status is not OrchestrationStepStatus.COMPLETED:
            raise ValueError(
                f"Orchestration step {current.step_id!r} must be COMPLETED before advancing."
            )

        next_index = self.current_step_index + 1

        if next_index >= len(self.steps):
            self.current_step_index = None
            return None

        return self.start_step(next_index)

    def _resolve_step_index(self, step_index: int | None) -> int:
        index = self.current_step_index if step_index is None else step_index

        if index is None:
            raise ValueError("Orchestration has no current step.")

        self._validate_step_index(index)
        return index

    def _validate_step_index(self, step_index: int) -> None:
        if not isinstance(step_index, int) or isinstance(step_index, bool):
            raise TypeError("Orchestration step_index must be an integer.")

        if step_index < 0 or step_index >= len(self.steps):
            raise IndexError(
                f"Orchestration step_index {step_index} is outside the available steps."
            )
