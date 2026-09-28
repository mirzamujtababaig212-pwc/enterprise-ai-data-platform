from __future__ import annotations

from dataclasses import dataclass

from ai_platform.agents.models import AgentResponse


@dataclass(frozen=True)
class AgentRuntimeEvaluationSnapshot:
    """
    Immutable observation of one completed semantic runtime iteration.

    The snapshot carries only runtime-boundary information. It is distinct
    from durable AgentRunEvidence, which represents consolidated post-run
    evaluation evidence.
    """

    iteration: int
    step_index: int | None
    response: AgentResponse
    tool_rounds: int

    def __post_init__(self) -> None:
        if not isinstance(self.iteration, int) or isinstance(self.iteration, bool):
            raise TypeError("Runtime evaluation iteration must be an integer.")

        if self.iteration <= 0:
            raise ValueError("Runtime evaluation iteration must be greater than zero.")

        if self.step_index is not None:
            if not isinstance(self.step_index, int) or isinstance(self.step_index, bool):
                raise TypeError("Runtime evaluation step_index must be an integer or None.")

            if self.step_index < 0:
                raise ValueError("Runtime evaluation step_index must not be negative.")

        if not isinstance(self.response, AgentResponse):
            raise TypeError("Runtime evaluation response must be an AgentResponse.")

        if not isinstance(self.tool_rounds, int) or isinstance(self.tool_rounds, bool):
            raise TypeError("Runtime evaluation tool_rounds must be an integer.")

        if self.tool_rounds < 0:
            raise ValueError("Runtime evaluation tool_rounds must not be negative.")
