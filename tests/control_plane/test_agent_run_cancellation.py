from __future__ import annotations

import asyncio

import pytest

from app.control_plane.agent_runs.cancellation import (
    AgentRunCancellationRegistry,
)


@pytest.mark.asyncio
async def test_register_and_cancel_active_task() -> None:
    registry = AgentRunCancellationRegistry()

    started = asyncio.Event()

    async def worker() -> None:
        started.set()
        await asyncio.Future()

    task = asyncio.create_task(worker())

    try:
        await started.wait()

        registry.register("run-123", task)

        assert registry.cancel("run-123") is True

        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        if not task.done():
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task


@pytest.mark.asyncio
async def test_cancel_returns_false_for_unknown_run() -> None:
    registry = AgentRunCancellationRegistry()

    assert registry.cancel("missing-run") is False


@pytest.mark.asyncio
async def test_cancel_returns_false_for_completed_task() -> None:
    registry = AgentRunCancellationRegistry()

    async def worker() -> None:
        return None

    task = asyncio.create_task(worker())
    await task

    registry.register("run-123", task)

    assert registry.cancel("run-123") is False


@pytest.mark.asyncio
async def test_unregister_only_removes_matching_task() -> None:
    registry = AgentRunCancellationRegistry()

    async def worker() -> None:
        await asyncio.Future()

    first_task = asyncio.create_task(worker())
    second_task = asyncio.create_task(worker())

    try:
        registry.register("run-123", first_task)
        registry.register("run-123", second_task)

        registry.unregister("run-123", first_task)

        assert registry.cancel("run-123") is True

        with pytest.raises(asyncio.CancelledError):
            await second_task

        assert not first_task.done()
    finally:
        if not first_task.done():
            first_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first_task
