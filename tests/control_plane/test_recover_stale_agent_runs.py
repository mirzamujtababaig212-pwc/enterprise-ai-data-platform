from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import scripts.recover_stale_agent_runs as cli
from app.control_plane.agent_runs.recovery_service import AgentRunRecoverySweepResult


class FakeSession:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_recover_stale_runs_closes_session_and_returns_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    db = FakeSession()
    recovery_service = AsyncMock()
    worker = AsyncMock()
    worker.run_once.return_value = AgentRunRecoverySweepResult(
        recovered=(
            SimpleNamespace(run_id="run-1"),
            SimpleNamespace(run_id="run-2"),
        ),
        failed_run_ids=(),
    )

    monkeypatch.setattr(cli, "SessionLocal", lambda: db)
    monkeypatch.setattr(
        cli,
        "build_agent_run_recovery_service",
        AsyncMock(return_value=recovery_service),
    )
    monkeypatch.setattr(
        cli,
        "AgentRunRecoveryWorker",
        Mock(return_value=worker),
    )

    result = await cli.recover_stale_runs(25)

    assert result == 0
    assert db.closed is True
    worker.run_once.assert_awaited_once()

    output = capsys.readouterr().out
    assert output == "Recovered 2 stale agent run(s).\n"


@pytest.mark.asyncio
async def test_recover_stale_runs_returns_one_when_sweep_has_failed_runs(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    db = FakeSession()
    recovery_service = AsyncMock()
    worker = AsyncMock()
    worker.run_once.return_value = AgentRunRecoverySweepResult(
        recovered=(SimpleNamespace(run_id="run-1"),),
        failed_run_ids=("run-2",),
    )

    monkeypatch.setattr(cli, "SessionLocal", lambda: db)
    monkeypatch.setattr(
        cli,
        "build_agent_run_recovery_service",
        AsyncMock(return_value=recovery_service),
    )
    monkeypatch.setattr(
        cli,
        "AgentRunRecoveryWorker",
        Mock(return_value=worker),
    )

    result = await cli.recover_stale_runs(25)

    assert result == 1
    assert db.closed is True
    worker.run_once.assert_awaited_once()

    captured = capsys.readouterr()
    assert captured.out == "Recovered 1 stale agent run(s).\n"
    assert captured.err == "Failed to recover 1 stale agent run(s): run-2\n"


@pytest.mark.asyncio
async def test_recover_stale_runs_closes_session_when_worker_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = FakeSession()
    recovery_service = AsyncMock()
    worker = AsyncMock()
    worker.run_once.side_effect = RuntimeError("database unavailable")

    monkeypatch.setattr(cli, "SessionLocal", lambda: db)
    monkeypatch.setattr(
        cli,
        "build_agent_run_recovery_service",
        AsyncMock(return_value=recovery_service),
    )
    monkeypatch.setattr(
        cli,
        "AgentRunRecoveryWorker",
        Mock(return_value=worker),
    )

    with pytest.raises(RuntimeError, match="database unavailable"):
        await cli.recover_stale_runs(25)

    assert db.closed is True


def test_main_rejects_non_positive_limit(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        cli,
        "parse_args",
        lambda: SimpleNamespace(limit=0),
    )

    result = cli.main()

    assert result == 2
    assert capsys.readouterr().err == "error: --limit must be greater than zero.\n"


def test_main_returns_one_for_unexpected_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        cli,
        "parse_args",
        lambda: SimpleNamespace(limit=25),
    )

    def fail(coro) -> int:
        coro.close()
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(
        cli.asyncio,
        "run",
        fail,
    )

    result = cli.main()

    assert result == 1
    assert (
        capsys.readouterr().err == "error: stale agent-run recovery failed: "
        "database unavailable\n"
    )
