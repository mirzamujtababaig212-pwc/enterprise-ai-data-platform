from __future__ import annotations

import argparse

from common.control.pipeline_control import PipelineControl
from common.spark.spark_builder import SparkSessionBuilder


def main() -> int:

    parser = argparse.ArgumentParser(description="Recover a stale RUNNING pipeline execution.")

    parser.add_argument(
        "--run-id",
        required=True,
        help="Run ID to recover.",
    )

    parser.add_argument(
        "--reason",
        default=("Recovered stale RUNNING execution " "during control-plane maintenance."),
        help="Failure reason recorded in control plane.",
    )

    args = parser.parse_args()

    spark = SparkSessionBuilder.build("RecoverStalePipelineRun")

    try:

        control = PipelineControl(spark)

        control.initialize()

        print("=" * 80)
        print("STALE RUN RECOVERY")
        print("=" * 80)

        print(f"Run ID : {args.run_id}")

        print(f"Reason : {args.reason}")

        print("=" * 80)

        control.recover_stale_run(
            run_id=args.run_id,
            reason=args.reason,
        )

        print("STALE RUN RECOVERED SUCCESSFULLY")

        print("=" * 80)

        return 0

    finally:
        spark.stop()


if __name__ == "__main__":
    raise SystemExit(main())
