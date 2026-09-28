from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class OrchestrationStepCompletionPolicy(StrEnum):
    ON_AGENT_RESPONSE = "on_agent_response"
    ON_TOOL_RESULT = "on_tool_result"


class AgentRuntimePhase(StrEnum):
    """Semantic phase of one agent runtime loop iteration."""

    PLAN = "plan"
    ACT = "act"
    OBSERVE = "observe"
    EVALUATE = "evaluate"


class AgentRuntimeDecision(StrEnum):
    """Decision produced by the evaluation phase."""

    CONTINUE = "continue"
    STOP = "stop"


_RUNTIME_PHASE_TRANSITIONS: dict[AgentRuntimePhase, frozenset[AgentRuntimePhase]] = {
    AgentRuntimePhase.PLAN: frozenset({AgentRuntimePhase.ACT}),
    AgentRuntimePhase.ACT: frozenset({AgentRuntimePhase.OBSERVE}),
    AgentRuntimePhase.OBSERVE: frozenset({AgentRuntimePhase.EVALUATE}),
    AgentRuntimePhase.EVALUATE: frozenset(),
}


def validate_runtime_phase_transition(
    current: AgentRuntimePhase,
    target: AgentRuntimePhase,
) -> None:
    """Validate a phase transition in the Runtime V1 loop."""

    if target not in _RUNTIME_PHASE_TRANSITIONS[current]:
        raise ValueError(
            f"Invalid agent runtime phase transition: " f"{current.value} -> {target.value}"
        )


def validate_runtime_decision(
    phase: AgentRuntimePhase,
    decision: AgentRuntimeDecision,
) -> None:
    """Validate that a runtime decision is emitted only from EVALUATE."""

    if phase is not AgentRuntimePhase.EVALUATE:
        raise ValueError(
            f"Agent runtime decision {decision.value!r} is only valid "
            f"from the evaluate phase; current phase is {phase.value!r}"
        )


@dataclass
class AgentRuntimeState:
    """
    Typed semantic state for one Runtime V1 loop.

    This state is deliberately separate from OrchestrationState:
    OrchestrationState owns logical step lifecycle, while this object owns
    the semantic Plan -> Act -> Observe -> Evaluate loop.
    """

    phase: AgentRuntimePhase = AgentRuntimePhase.PLAN
    decision: AgentRuntimeDecision | None = None
    current_step_index: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.phase, AgentRuntimePhase):
            raise TypeError("Agent runtime phase must be an AgentRuntimePhase.")

        if self.decision is not None and not isinstance(
            self.decision,
            AgentRuntimeDecision,
        ):
            raise TypeError("Agent runtime decision must be an AgentRuntimeDecision or None.")

        if self.current_step_index is not None:
            if not isinstance(self.current_step_index, int) or isinstance(
                self.current_step_index,
                bool,
            ):
                raise TypeError("Agent runtime current_step_index must be an integer.")

            if self.current_step_index < 0:
                raise ValueError("Agent runtime current_step_index must not be negative.")

        if self.phase is not AgentRuntimePhase.EVALUATE and self.decision is not None:
            raise ValueError("Agent runtime decision must be None before the evaluate phase.")

    def transition_to(
        self,
        target: AgentRuntimePhase,
        *,
        current_step_index: int | None = None,
    ) -> None:
        """Advance the runtime to the next valid semantic phase."""

        if not isinstance(target, AgentRuntimePhase):
            raise TypeError("Agent runtime target phase must be an AgentRuntimePhase.")

        validate_runtime_phase_transition(self.phase, target)

        if current_step_index is not None:
            if not isinstance(current_step_index, int) or isinstance(
                current_step_index,
                bool,
            ):
                raise TypeError("Agent runtime current_step_index must be an integer.")

            if current_step_index < 0:
                raise ValueError("Agent runtime current_step_index must not be negative.")

        self.phase = target
        self.decision = None
        self.current_step_index = current_step_index

    def evaluate(
        self,
        decision: AgentRuntimeDecision,
    ) -> None:
        """Record the decision produced by the evaluation phase."""

        if not isinstance(decision, AgentRuntimeDecision):
            raise TypeError("Agent runtime decision must be an AgentRuntimeDecision.")

        validate_runtime_decision(self.phase, decision)
        self.decision = decision

    def continue_to_plan(
        self,
        *,
        current_step_index: int | None = None,
    ) -> None:
        """
        Apply a CONTINUE decision and begin the next loop at PLAN.
        """

        if self.phase is not AgentRuntimePhase.EVALUATE:
            raise ValueError("Agent runtime can continue only from the evaluate phase.")

        if self.decision is not AgentRuntimeDecision.CONTINUE:
            raise ValueError(
                "Agent runtime must have a CONTINUE decision before returning to plan."
            )

        if current_step_index is not None:
            if not isinstance(current_step_index, int) or isinstance(
                current_step_index,
                bool,
            ):
                raise TypeError("Agent runtime current_step_index must be an integer.")

            if current_step_index < 0:
                raise ValueError("Agent runtime current_step_index must not be negative.")

        self.phase = AgentRuntimePhase.PLAN
        self.decision = None
        self.current_step_index = current_step_index

    def stop(self) -> None:
        """Validate and retain a terminal STOP decision."""

        if self.phase is not AgentRuntimePhase.EVALUATE:
            raise ValueError("Agent runtime can stop only from the evaluate phase.")

        if self.decision is not AgentRuntimeDecision.STOP:
            raise ValueError("Agent runtime must have a STOP decision before stopping.")

    def to_metadata(self) -> dict[str, Any]:
        """Serialize runtime state for durable agent-step metadata."""

        return {
            "runtime": {
                "phase": self.phase.value,
                "decision": (self.decision.value if self.decision is not None else None),
                "current_step_index": self.current_step_index,
            }
        }

    @classmethod
    def from_metadata(
        cls,
        metadata: dict[str, Any],
    ) -> "AgentRuntimeState | None":
        """Restore runtime state from durable agent-step metadata."""

        if not isinstance(metadata, dict):
            raise TypeError("Agent runtime metadata must be a dictionary.")

        runtime_metadata = metadata.get("runtime")

        if runtime_metadata is None:
            return None

        if not isinstance(runtime_metadata, dict):
            raise ValueError("Agent runtime metadata must contain a dictionary under 'runtime'.")

        if "phase" not in runtime_metadata:
            raise ValueError("Agent runtime metadata is missing 'phase'.")

        try:
            phase = AgentRuntimePhase(runtime_metadata["phase"])
        except (TypeError, ValueError) as exc:
            raise ValueError("Agent runtime metadata contains an invalid phase.") from exc

        decision_value = runtime_metadata.get("decision")
        if decision_value is None:
            decision = None
        else:
            try:
                decision = AgentRuntimeDecision(decision_value)
            except (TypeError, ValueError) as exc:
                raise ValueError("Agent runtime metadata contains an invalid decision.") from exc

        current_step_index = runtime_metadata.get("current_step_index")

        return cls(
            phase=phase,
            decision=decision,
            current_step_index=current_step_index,
        )


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


@dataclass(frozen=True)
class OrchestrationStepResult:
    """
    Durable-in-memory result produced by one logical orchestration step.

    The result is intentionally provider- and domain-neutral. Lifecycle
    transitions remain owned by OrchestrationState and storing a result
    does not change step status.
    """

    step_id: str
    output: Any
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.step_id, str) or not self.step_id.strip():
            raise ValueError("Orchestration step result step_id must not be empty.")

        if not isinstance(self.metadata, dict):
            raise TypeError("Orchestration step result metadata must be a dictionary.")

        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass
class OrchestrationState:
    """
    Mutable execution state for logical orchestration progression.
    """

    steps: list[OrchestrationStep] = field(default_factory=list)
    current_step_index: int | None = None
    step_results: dict[str, OrchestrationStepResult] = field(default_factory=dict)

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

        self.step_results = dict(self.step_results)

        step_ids = {step.step_id for step in self.steps}

        for step_id, result in self.step_results.items():
            if not isinstance(step_id, str) or not step_id.strip():
                raise ValueError("Orchestration step result keys must not be empty.")

            if step_id not in step_ids:
                raise ValueError(
                    f"Orchestration step result references unknown step_id {step_id!r}."
                )

            if not isinstance(result, OrchestrationStepResult):
                raise TypeError(
                    "Orchestration step_results must contain " "OrchestrationStepResult instances."
                )

            if result.step_id != step_id:
                raise ValueError("Orchestration step result key must match result.step_id.")

    @property
    def current_step(self) -> OrchestrationStep | None:
        """Return the current logical step, if one is selected."""
        if self.current_step_index is None:
            return None

        return self.steps[self.current_step_index]

    def set_step_result(
        self,
        step_id: str,
        output: Any,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> OrchestrationStepResult:
        """
        Store the result produced by a logical orchestration step.

        Storing a result is deliberately independent from lifecycle status.
        Callers must explicitly complete or fail the step.
        """
        index = self._resolve_step_id(step_id)
        resolved_step_id = self.steps[index].step_id

        result = OrchestrationStepResult(
            step_id=resolved_step_id,
            output=output,
            metadata={} if metadata is None else metadata,
        )

        self.step_results[resolved_step_id] = result

        return result

    def get_step_result(
        self,
        step_id: str,
    ) -> OrchestrationStepResult | None:
        """Return the stored result for a logical step, if one exists."""
        index = self._resolve_step_id(step_id)
        resolved_step_id = self.steps[index].step_id
        return self.step_results.get(resolved_step_id)

    def get_completed_step_result(
        self,
        step_id: str,
    ) -> OrchestrationStepResult:
        """
        Return a logical step result only after the step has completed.

        Downstream orchestration stages must not consume a result from a
        still-running or otherwise incomplete step.
        """
        index = self._resolve_step_id(step_id)
        step = self.steps[index]

        if step.status is not OrchestrationStepStatus.COMPLETED:
            raise ValueError(
                f"Orchestration step {step.step_id!r} must be COMPLETED "
                "before its result can be consumed."
            )

        result = self.step_results.get(step.step_id)

        if result is None:
            raise ValueError(
                f"Orchestration step {step.step_id!r} is COMPLETED " "but has no stored result."
            )

        return result

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

    def _resolve_step_id(self, step_id: str) -> int:
        """Resolve a logical step ID to its state index."""
        if not isinstance(step_id, str) or not step_id.strip():
            raise ValueError("Orchestration step_id must not be empty.")

        for index, step in enumerate(self.steps):
            if step.step_id == step_id:
                return index

        raise ValueError(f"Unknown orchestration step {step_id!r}")

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
