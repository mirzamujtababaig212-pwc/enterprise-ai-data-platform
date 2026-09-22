from __future__ import annotations

import argparse
import asyncio
import sys

from app.control_plane.agent_runs.recovery_worker import (
    AgentRunRecoveryWorker,
)
from app.control_plane.dependencies import (
    build_agent_run_recovery_service,
)
from app.control_plane.persistence.database import SessionLocal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover stale durable agent runs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of stale runs to inspect in one sweep.",
    )
    return parser.parse_args()


async def recover_stale_runs(limit: int) -> int:
    db = SessionLocal()
    try:
        recovery_service = await build_agent_run_recovery_service(db)
        worker = AgentRunRecoveryWorker(
            recovery_service=recovery_service,
            limit=limit,
        )
        result = await worker.run_once()
        print(f"Recovered {len(result.recovered)} stale agent run(s).")

        if result.failed_run_ids:
            print(
                "Failed to recover "
                f"{len(result.failed_run_ids)} stale agent run(s): "
                f"{', '.join(result.failed_run_ids)}",
                file=sys.stderr,
            )
            return 1

        return 0
    finally:
        db.close()


def main() -> int:
    args = parse_args()

    if args.limit <= 0:
        print("error: --limit must be greater than zero.", file=sys.stderr)
        return 2

    try:
        return asyncio.run(recover_stale_runs(args.limit))
    except Exception as exc:
        print(
            f"error: stale agent-run recovery failed: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
