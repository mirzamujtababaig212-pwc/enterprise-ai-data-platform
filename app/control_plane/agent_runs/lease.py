from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from app.control_plane.agent_runs.repository import AgentRunRepository
from uuid import uuid4


def create_lease(lease_seconds: int) -> tuple[str, datetime]:
    if lease_seconds <= 0:
        raise ValueError("lease_seconds must be greater than zero.")

    lease_id = str(uuid4())
    lease_expires_at = datetime.now(UTC) + timedelta(seconds=lease_seconds)

    return lease_id, lease_expires_at


async def heartbeat_loop(
    repository: AgentRunRepository,
    *,
    run_id: str,
    lease_id: str,
    lease_seconds: int,
    ownership_lost: asyncio.Event | None = None,
) -> None:
    interval_seconds = lease_seconds / 3

    while True:
        await asyncio.sleep(interval_seconds)

        lease_expires_at = datetime.now(UTC) + timedelta(
            seconds=lease_seconds,
        )

        try:
            renewed_run = repository.heartbeat(
                run_id,
                lease_id=lease_id,
                lease_expires_at=lease_expires_at,
            )
        except Exception:
            # Heartbeat failures must not mask the agent execution
            # result. Ownership remains enforced by the repository's
            # conditional terminal transition.
            continue

        if renewed_run is None:
            # Another worker owns the run, or the run is no longer
            # RUNNING. Signal the active execution so it can stop
            # rather than continuing as a stale worker.
            if ownership_lost is not None:
                ownership_lost.set()
            return
