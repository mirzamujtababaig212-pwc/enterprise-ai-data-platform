from __future__ import annotations

import asyncio


class AgentRunCancellationRegistry:
    """
    Process-local coordinator for interrupting active agent-run tasks.

    Durable run state remains owned by AgentRunRepository. This registry only
    provides the in-process signal from a cancellation request to the active
    asyncio task.
    """

    def __init__(self) -> None:
        self._tasks: dict[str, asyncio.Task[object]] = {}

    def register(self, run_id: str, task: asyncio.Task[object]) -> None:
        self._tasks[run_id] = task

    def unregister(self, run_id: str, task: asyncio.Task[object]) -> None:
        if self._tasks.get(run_id) is task:
            self._tasks.pop(run_id, None)

    def cancel(self, run_id: str) -> bool:
        task = self._tasks.get(run_id)

        if task is None or task.done():
            return False

        return task.cancel()
