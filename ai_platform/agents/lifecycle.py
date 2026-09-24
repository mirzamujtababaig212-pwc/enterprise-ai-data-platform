from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import StrEnum


class AgentExecutionLifecyclePhase(StrEnum):
    """
    Runtime lifecycle phases for one agent execution.
    """

    CREATED = "created"
    PREPARING = "preparing"
    PRE_EXECUTION = "pre_execution"
    ORCHESTRATING = "orchestrating"
    EXECUTING = "executing"
    POST_EXECUTION = "post_execution"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class AgentExecutionLifecycleTransition:
    """
    Immutable record of one lifecycle phase transition.
    """

    sequence: int
    from_phase: AgentExecutionLifecyclePhase
    to_phase: AgentExecutionLifecyclePhase


@dataclass(frozen=True)
class AgentExecutionLifecycleSnapshot:
    """
    Immutable point-in-time view of lifecycle state.
    """

    phase: AgentExecutionLifecyclePhase
    sequence: int
    transitions: tuple[AgentExecutionLifecycleTransition, ...]


@dataclass
class AgentExecutionLifecycleState:
    """
    Concurrency-safe, per-execution lifecycle coordination state.

    This state is intentionally separate from durable orchestration,
    execution-budget state, checkpoint state, and recovery state. Those
    existing components remain authoritative for their respective concerns.

    The lifecycle state only records which runtime phase this invocation
    currently occupies and the ordered transitions that occurred.
    """

    phase: AgentExecutionLifecyclePhase = AgentExecutionLifecyclePhase.CREATED
    _sequence: int = field(default=0, init=False, repr=False)
    _transitions: list[AgentExecutionLifecycleTransition] = field(
        default_factory=list,
        init=False,
        repr=False,
    )
    _lock: asyncio.Lock = field(
        default_factory=asyncio.Lock,
        init=False,
        repr=False,
    )

    async def transition(
        self,
        to_phase: AgentExecutionLifecyclePhase,
        *,
        expected_phase: AgentExecutionLifecyclePhase | None = None,
    ) -> AgentExecutionLifecycleTransition:
        """
        Atomically transition to a new lifecycle phase.

        When expected_phase is supplied, the transition succeeds only if the
        current phase still matches it. This prevents concurrent lifecycle
        participants from silently overwriting each other's transitions.
        """
        if not isinstance(to_phase, AgentExecutionLifecyclePhase):
            raise TypeError("Lifecycle transition target must be an AgentExecutionLifecyclePhase.")

        if expected_phase is not None and not isinstance(
            expected_phase,
            AgentExecutionLifecyclePhase,
        ):
            raise TypeError(
                "Lifecycle expected_phase must be an AgentExecutionLifecyclePhase or None."
            )

        async with self._lock:
            if expected_phase is not None and self.phase is not expected_phase:
                raise RuntimeError(
                    "Lifecycle transition expected phase "
                    f"{expected_phase.value!r}, but current phase is {self.phase.value!r}."
                )

            if self.phase is to_phase:
                raise RuntimeError(f"Lifecycle phase is already {to_phase.value!r}.")

            transition = AgentExecutionLifecycleTransition(
                sequence=self._sequence + 1,
                from_phase=self.phase,
                to_phase=to_phase,
            )

            self._sequence = transition.sequence
            self.phase = to_phase
            self._transitions.append(transition)

            return transition

    async def snapshot(self) -> AgentExecutionLifecycleSnapshot:
        """
        Return an immutable snapshot of the current lifecycle state.
        """
        async with self._lock:
            return AgentExecutionLifecycleSnapshot(
                phase=self.phase,
                sequence=self._sequence,
                transitions=tuple(self._transitions),
            )
