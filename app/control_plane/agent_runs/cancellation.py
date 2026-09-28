from __future__ import annotations

import asyncio
from dataclasses import dataclass


@dataclass
class _CancellationRegistration:
    task: asyncio.Task[object]
    cancellation_requested: asyncio.Event


class AgentRunCancellationRegistry:
    """
    Process-local coordinator for interrupting active agent-run tasks.

    Durable run state remains owned by AgentRunRepository. This registry only
    provides the in-process signal from a cancellation request to the active
    asyncio task.
    """

    def __init__(self) -> None:
        self._registrations: dict[str, _CancellationRegistration] = {}

    def register(
        self,
        run_id: str,
        task: asyncio.Task[object],
        cancellation_requested: asyncio.Event,
    ) -> None:
        self._registrations[run_id] = _CancellationRegistration(
            task=task,
            cancellation_requested=cancellation_requested,
        )

    def unregister(self, run_id: str, task: asyncio.Task[object]) -> None:
        registration = self._registrations.get(run_id)

        if registration is not None and registration.task is task:
            self._registrations.pop(run_id, None)

    def cancel(self, run_id: str) -> bool:
        registration = self._registrations.get(run_id)

        if registration is None or registration.task.done():
            return False

        registration.cancellation_requested.set()
        return True
