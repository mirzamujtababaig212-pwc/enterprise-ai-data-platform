from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


COMMANDS = [
    [
        sys.executable,
        str(PROJECT_ROOT / "spark" / "bronze" / "ingest_vehicle_data.py"),
    ],
    [
        sys.executable,
        str(PROJECT_ROOT / "spark" / "pipelines" / "bronze_to_silver_pipeline.py"),
    ],
    [
        sys.executable,
        str(PROJECT_ROOT / "spark" / "pipelines" / "silver_to_gold_pipeline.py"),
    ],
    [
        sys.executable,
        str(PROJECT_ROOT / "tests" / "data_quality" / "test_vehicle_data_quality.py"),
    ],
    [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "verify_gold_table.py"),
    ],
]


def run_command(command: list[str]) -> None:
    print()
    print("=" * 80)
    print("RUNNING")
    print("=" * 80)
    print(" ".join(command))
    print()

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
    )

    if result.returncode != 0:
        raise SystemExit(f"Command failed with exit code " f"{result.returncode}")


def main() -> None:
    print("=" * 80)
    print("ENTERPRISE AI PLATFORM")
    print("END-TO-END DATA PIPELINE")
    print("=" * 80)

    for command in COMMANDS:
        run_command(command)

    print()
    print("=" * 80)
    print("PIPELINE EXECUTION SUCCESSFUL")
    print("=" * 80)


if __name__ == "__main__":
    main()
