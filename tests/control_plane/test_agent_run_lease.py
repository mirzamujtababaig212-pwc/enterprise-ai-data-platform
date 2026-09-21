import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from app.control_plane.agent_runs.lease import create_lease, heartbeat_loop


def test_create_lease_returns_id_and_future_expiry() -> None:
    before = datetime.now(UTC)

    lease_id, lease_expires_at = create_lease(60)

    after = datetime.now(UTC)

    assert lease_id
    assert lease_expires_at.tzinfo == UTC
    assert lease_expires_at > before
    assert lease_expires_at <= after + timedelta(seconds=60)


@pytest.mark.parametrize("lease_seconds", [0, -1])
def test_create_lease_rejects_non_positive_duration(
    lease_seconds: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="lease_seconds must be greater than zero",
    ):
        create_lease(lease_seconds)


class FakeRepository:
    def __init__(self, heartbeat_results):
        self.heartbeat_results = list(heartbeat_results)
        self.heartbeat_calls = []

    def heartbeat(self, run_id, *, lease_id, lease_expires_at):
        self.heartbeat_calls.append(
            {
                "run_id": run_id,
                "lease_id": lease_id,
                "lease_expires_at": lease_expires_at,
            }
        )
        return self.heartbeat_results.pop(0)


@pytest.mark.asyncio
async def test_heartbeat_loop_signals_ownership_loss() -> None:
    repository = FakeRepository([None])
    ownership_lost = asyncio.Event()

    task = asyncio.create_task(
        heartbeat_loop(
            repository,
            run_id="run-123",
            lease_id="lease-123",
            lease_seconds=0.01,
            ownership_lost=ownership_lost,
        )
    )

    await task

    assert ownership_lost.is_set()
    assert len(repository.heartbeat_calls) == 1


@pytest.mark.asyncio
async def test_heartbeat_loop_does_not_signal_on_persistence_failure() -> None:
    class FailingRepository:
        def __init__(self) -> None:
            self.calls = 0

        def heartbeat(self, run_id, *, lease_id, lease_expires_at):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary persistence failure")
            return None

    repository = FailingRepository()
    ownership_lost = asyncio.Event()

    task = asyncio.create_task(
        heartbeat_loop(
            repository,
            run_id="run-123",
            lease_id="lease-123",
            lease_seconds=0.01,
            ownership_lost=ownership_lost,
        )
    )

    await task

    assert repository.calls == 2
    assert ownership_lost.is_set()


@pytest.mark.asyncio
async def test_heartbeat_loop_without_signal_event_still_stops_on_lost_ownership() -> None:
    repository = FakeRepository([None])

    await heartbeat_loop(
        repository,
        run_id="run-123",
        lease_id="lease-123",
        lease_seconds=0.01,
    )

    assert len(repository.heartbeat_calls) == 1
