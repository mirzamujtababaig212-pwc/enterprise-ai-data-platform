from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from common.control.pipeline_control import PipelineControl
from common.spark.spark_builder import SparkSessionBuilder


def utc_now() -> datetime:
    return datetime.now(UTC)


def parse_optional_int(
    value: str | None,
) -> int | None:

    if value is None:
        return None

    return int(value)


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(description="Enterprise AI Platform control-plane worker")

    subparsers = parser.add_subparsers(
        dest="action",
        required=True,
    )

    # --------------------------------------------------------------
    # START
    # --------------------------------------------------------------

    start_parser = subparsers.add_parser("start")

    start_parser.add_argument(
        "--run-id",
        required=True,
    )

    start_parser.add_argument(
        "--pipeline-name",
        required=True,
    )

    # --------------------------------------------------------------
    # RECORD STAGE
    # --------------------------------------------------------------

    stage_parser = subparsers.add_parser("record-stage")

    stage_parser.add_argument(
        "--run-id",
        required=True,
    )

    stage_parser.add_argument(
        "--pipeline-name",
        required=True,
    )

    stage_parser.add_argument(
        "--stage-name",
        required=True,
    )

    stage_parser.add_argument(
        "--status",
        required=True,
        choices=[
            "SUCCESS",
            "FAILED",
        ],
    )

    stage_parser.add_argument(
        "--start-time",
        required=True,
    )

    stage_parser.add_argument(
        "--end-time",
        required=True,
    )

    stage_parser.add_argument(
        "--input-rows",
    )

    stage_parser.add_argument(
        "--output-rows",
    )

    stage_parser.add_argument(
        "--dq-checks",
    )

    stage_parser.add_argument(
        "--dq-failures",
    )

    stage_parser.add_argument(
        "--error-message",
        default=None,
    )

    # --------------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------------

    complete_parser = subparsers.add_parser("complete")

    complete_parser.add_argument(
        "--run-id",
        required=True,
    )

    complete_parser.add_argument(
        "--pipeline-name",
        required=True,
    )

    complete_parser.add_argument(
        "--status",
        required=True,
        choices=[
            "SUCCESS",
            "FAILED",
        ],
    )

    complete_parser.add_argument(
        "--start-time",
        required=True,
    )

    complete_parser.add_argument(
        "--end-time",
        required=True,
    )

    complete_parser.add_argument(
        "--error-message",
        default=None,
    )

    return parser


def parse_datetime(
    value: str,
) -> datetime:

    return datetime.fromisoformat(value)


def main() -> int:

    parser = build_parser()
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]

    print("=" * 80)
    print("CONTROL PLANE ACTION")
    print("=" * 80)
    print(f"Action       : {args.action}")
    print(f"Project root : {project_root}")
    print("=" * 80)

    spark = None

    try:

        spark = SparkSessionBuilder.build("EnterpriseControlPlaneAction")

        control = PipelineControl(spark)

        control.initialize()

        # ----------------------------------------------------------
        # START
        # ----------------------------------------------------------

        if args.action == "start":

            start_time = control.start_pipeline_run(
                run_id=args.run_id,
                pipeline_name=args.pipeline_name,
            )

            print(f"CONTROL START SUCCESS | " f"run_id={args.run_id}")

            print(f"Start time: " f"{start_time.isoformat()}")

            return 0

        # ----------------------------------------------------------
        # RECORD STAGE
        # ----------------------------------------------------------

        if args.action == "record-stage":

            control.record_stage(
                run_id=args.run_id,
                pipeline_name=args.pipeline_name,
                stage_name=args.stage_name,
                status=args.status,
                start_time=parse_datetime(args.start_time),
                end_time=parse_datetime(args.end_time),
                input_rows=parse_optional_int(args.input_rows),
                output_rows=parse_optional_int(args.output_rows),
                dq_checks=parse_optional_int(args.dq_checks),
                dq_failures=parse_optional_int(args.dq_failures),
                error_message=args.error_message,
            )

            print(
                f"CONTROL STAGE SUCCESS | "
                f"run_id={args.run_id} | "
                f"stage={args.stage_name} | "
                f"status={args.status}"
            )

            return 0

        # ----------------------------------------------------------
        # COMPLETE
        # ----------------------------------------------------------

        if args.action == "complete":

            control.complete_pipeline_run(
                run_id=args.run_id,
                pipeline_name=args.pipeline_name,
                status=args.status,
                start_time=parse_datetime(args.start_time),
                end_time=parse_datetime(args.end_time),
                error_message=args.error_message,
            )

            print(f"CONTROL COMPLETE SUCCESS | " f"run_id={args.run_id} | " f"status={args.status}")

            return 0

        raise RuntimeError(f"Unsupported action: {args.action}")

    except Exception as exc:

        print(f"CONTROL PLANE ACTION FAILED: {exc}")

        return 1

    finally:

        if spark is not None:
            spark.stop()


if __name__ == "__main__":
    raise SystemExit(main())
