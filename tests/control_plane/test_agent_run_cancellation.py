from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from ai_platform.agents.execution import AgentExecutionContext
from ai_platform.agents.models import AgentRequest
from app.control_plane.agent_runs.cancellation import AgentRunCancellationRegistry


@pytest.mark.asyncio
async def test_register_and_cancel_active_task() -> None:
    registry = AgentRunCancellationRegistry()

    started = asyncio.Event()
    cancellation_requested = asyncio.Event()

    async def worker() -> None:
        started.set()
        await cancellation_requested.wait()
        raise asyncio.CancelledError("Agent execution cancellation requested.")

    task = asyncio.create_task(worker())

    try:
        await started.wait()

        registry.register(
            "run-123",
            task,
            cancellation_requested,
        )

        assert registry.cancel("run-123") is True
        assert cancellation_requested.is_set()

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
    cancellation_requested = asyncio.Event()

    async def worker() -> None:
        return None

    task = asyncio.create_task(worker())
    await task

    registry.register(
        "run-123",
        task,
        cancellation_requested,
    )

    assert registry.cancel("run-123") is False
    assert not cancellation_requested.is_set()


@pytest.mark.asyncio
async def test_unregister_only_removes_matching_task() -> None:
    registry = AgentRunCancellationRegistry()

    first_started = asyncio.Event()
    second_started = asyncio.Event()
    first_cancellation_requested = asyncio.Event()
    second_cancellation_requested = asyncio.Event()

    async def first_worker() -> None:
        first_started.set()
        await first_cancellation_requested.wait()
        raise asyncio.CancelledError("First task cancelled.")

    async def second_worker() -> None:
        second_started.set()
        await second_cancellation_requested.wait()
        raise asyncio.CancelledError("Second task cancelled.")

    first_task = asyncio.create_task(first_worker())
    second_task = asyncio.create_task(second_worker())

    try:
        await first_started.wait()
        await second_started.wait()

        registry.register(
            "run-123",
            first_task,
            first_cancellation_requested,
        )
        registry.register(
            "run-123",
            second_task,
            second_cancellation_requested,
        )

        registry.unregister("run-123", first_task)

        assert registry.cancel("run-123") is True
        assert second_cancellation_requested.is_set()
        assert not first_cancellation_requested.is_set()

        with pytest.raises(asyncio.CancelledError):
            await second_task

        assert not first_task.done()
    finally:
        if not first_task.done():
            first_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first_task


@pytest.mark.asyncio
async def test_cancellation_signal_reaches_execution_context() -> None:
    registry = AgentRunCancellationRegistry()
    cancellation_requested = asyncio.Event()
    started = asyncio.Event()
    cancellation_observed = asyncio.Event()

    async def worker() -> None:
        started.set()

        await cancellation_requested.wait()

        context = AgentExecutionContext(
            AgentRequest(input="Test cancellation"),
            tools=MagicMock(),
            llm=MagicMock(),
            cancellation_requested=cancellation_requested,
        )

        try:
            context.raise_if_cancellation_requested()
        except asyncio.CancelledError:
            cancellation_observed.set()
            raise

    task = asyncio.create_task(worker())

    try:
        await started.wait()

        registry.register(
            "run-123",
            task,
            cancellation_requested,
        )

        assert registry.cancel("run-123") is True
        assert cancellation_requested.is_set()

        with pytest.raises(asyncio.CancelledError):
            await task

        assert cancellation_observed.is_set()
    finally:
        if not task.done():
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
