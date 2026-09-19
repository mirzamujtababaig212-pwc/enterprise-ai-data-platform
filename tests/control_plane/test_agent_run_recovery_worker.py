from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.control_plane.agent_runs.models import AgentRunExecutionResult
from app.control_plane.agent_runs.recovery_worker import (
    AgentRunRecoveryWorker,
)


def test_worker_rejects_non_positive_limit() -> None:
    recovery_service = AsyncMock()

    with pytest.raises(
        ValueError,
        match="Recovery worker limit must be greater than zero.",
    ):
        AgentRunRecoveryWorker(
            recovery_service=recovery_service,
            limit=0,
        )


@pytest.mark.asyncio
async def test_run_once_delegates_to_recovery_service() -> None:
    recovery_service = AsyncMock()

    expected = [
        AgentRunExecutionResult(
            run_id="run-1",
            response={"output": "recovered"},
        )
    ]
    recovery_service.recover_stale_runs.return_value = expected

    worker = AgentRunRecoveryWorker(
        recovery_service=recovery_service,
        limit=25,
    )

    result = await worker.run_once()

    assert result == expected
    recovery_service.recover_stale_runs.assert_awaited_once_with(
        limit=25,
    )


@pytest.mark.asyncio
async def test_run_once_returns_empty_result_when_nothing_is_recovered() -> None:
    recovery_service = AsyncMock()
    recovery_service.recover_stale_runs.return_value = []

    worker = AgentRunRecoveryWorker(
        recovery_service=recovery_service,
        limit=10,
    )

    result = await worker.run_once()

    assert result == []
    recovery_service.recover_stale_runs.assert_awaited_once_with(
        limit=10,
    )
