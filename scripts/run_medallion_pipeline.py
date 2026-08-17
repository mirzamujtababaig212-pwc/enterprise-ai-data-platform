from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from common.control.pipeline_control import PipelineControl
from common.spark.spark_builder import SparkSessionBuilder

PIPELINE_NAME = "enterprise_medallion_pipeline"


@dataclass(frozen=True)
class PipelineStage:
    name: str
    script: Path


def utc_now() -> datetime:
    """
    Return timezone-aware UTC datetime.
    """

    return datetime.now(UTC)


def build_logger() -> logging.Logger:
    """
    Configure orchestrator logger.
    """

    logger = logging.getLogger("medallion_orchestrator")

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | " "%(name)s | %(message)s")

    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger


def parse_args() -> argparse.Namespace:
    """
    Parse orchestration arguments.
    """

    parser = argparse.ArgumentParser(description=("Enterprise AI Platform " "Medallion Pipeline"))

    parser.add_argument(
        "--retry-of",
        dest="retry_of",
        default=None,
        help=("Parent run ID when this execution " "is a retry."),
    )

    parser.add_argument(
        "--retry-attempt",
        dest="retry_attempt",
        type=int,
        default=0,
        help=("Retry attempt number. " "Initial execution is 0."),
    )

    parser.add_argument(
        "--allow-active-run",
        action="store_true",
        help=("Allow execution even if another " "pipeline run is currently RUNNING."),
    )

    return parser.parse_args()


def build_stages(
    project_root: Path,
) -> list[PipelineStage]:
    """
    Define ordered medallion pipeline stages.
    """

    return [
        PipelineStage(
            name="bronze_to_silver",
            script=(project_root / "spark" / "pipelines" / "bronze_to_silver_pipeline.py"),
        ),
        PipelineStage(
            name="silver_to_gold",
            script=(project_root / "spark" / "pipelines" / "silver_to_gold_pipeline.py"),
        ),
    ]


def table_exists(
    spark,
    table_name: str,
) -> bool:
    """
    Safely determine whether a Spark table exists.
    """

    try:
        return spark.catalog.tableExists(table_name)
    except Exception:
        return False


def get_table_row_count(
    spark,
    table_name: str,
) -> int | None:
    """
    Return current row count.

    Returns None if unavailable.
    """

    if not table_exists(
        spark,
        table_name,
    ):
        return None

    try:
        return int(spark.table(table_name).count())
    except Exception:
        return None


def collect_stage_metrics(
    spark,
    stage_name: str,
) -> tuple[
    int | None,
    int | None,
]:
    """
    Collect stage input/output row counts.
    """

    if stage_name == "bronze_to_silver":

        return (
            get_table_row_count(
                spark,
                "bronze.vehicle_events",
            ),
            get_table_row_count(
                spark,
                "silver.vehicle_events",
            ),
        )

    if stage_name == "silver_to_gold":

        return (
            get_table_row_count(
                spark,
                "silver.vehicle_events",
            ),
            get_table_row_count(
                spark,
                "gold.vehicle_metrics",
            ),
        )

    return None, None


def execute_stage(
    stage: PipelineStage,
    project_root: Path,
    logger: logging.Logger,
) -> tuple[
    int,
    datetime,
    datetime,
    str,
]:
    """
    Execute one pipeline stage.
    """

    start_time = utc_now()

    logger.info("=" * 80)
    logger.info(
        "STARTING STAGE: %s",
        stage.name,
    )
    logger.info(
        "SCRIPT: %s",
        stage.script,
    )
    logger.info(
        "START TIME: %s",
        start_time.isoformat(),
    )
    logger.info("=" * 80)

    if not stage.script.exists():

        error_message = "Stage script does not exist: " f"{stage.script}"

        logger.error(error_message)

        return (
            1,
            start_time,
            utc_now(),
            error_message,
        )

    command = [
        sys.executable,
        str(stage.script),
    ]

    logger.info(
        "Executing command: %s",
        " ".join(command),
    )

    process = subprocess.run(
        command,
        cwd=str(project_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=os.environ.copy(),
    )

    end_time = utc_now()

    combined_output = process.stdout or ""

    if combined_output:
        print(
            combined_output,
            end="",
        )

    duration_seconds = (end_time - start_time).total_seconds()

    if process.returncode == 0:

        logger.info(
            "STAGE COMPLETED: %s | " "duration=%.2f sec",
            stage.name,
            duration_seconds,
        )

    else:

        logger.error(
            "STAGE FAILED: %s | " "exit_code=%s | " "duration=%.2f sec",
            stage.name,
            process.returncode,
            duration_seconds,
        )

    logger.info(
        "END TIME: %s",
        end_time.isoformat(),
    )

    logger.info("=" * 80)

    return (
        process.returncode,
        start_time,
        end_time,
        combined_output,
    )


def main() -> int:

    args = parse_args()

    logger = build_logger()

    project_root = Path(__file__).resolve().parents[1]

    stages = build_stages(project_root)

    logger.info("=" * 80)
    logger.info("ENTERPRISE MEDALLION DATA PIPELINE")
    logger.info("=" * 80)

    logger.info(
        "Project root : %s",
        project_root,
    )

    logger.info(
        "Retry of     : %s",
        args.retry_of,
    )

    logger.info(
        "Retry attempt: %s",
        args.retry_attempt,
    )

    pipeline_wall_start = utc_now()

    logger.info(
        "Pipeline start: %s",
        pipeline_wall_start.isoformat(),
    )

    spark = None
    control = None

    run_id: str | None = None
    control_start_time: datetime | None = None

    try:

        # ==============================================================
        # CONTROL PLANE
        # ==============================================================

        logger.info("Initializing pipeline control plane...")

        spark = SparkSessionBuilder.build("EnterpriseMedallionPipelineControl")

        control = PipelineControl(spark)

        control.initialize()

        # --------------------------------------------------------------
        # ACTIVE RUN PROTECTION
        # --------------------------------------------------------------

        if not args.allow_active_run and control.has_active_run(PIPELINE_NAME):

            logger.error("Pipeline already has an " "active RUNNING execution.")

            logger.error("Use --allow-active-run only " "when intentional.")

            return 2

        # --------------------------------------------------------------
        # CREATE RUN
        # --------------------------------------------------------------

        run_id = control.generate_run_id()

        control_start_time = control.start_pipeline_run(
            run_id=run_id,
            pipeline_name=PIPELINE_NAME,
            parent_run_id=args.retry_of,
            retry_attempt=args.retry_attempt,
        )

        logger.info(
            "Control-plane run_id: %s",
            run_id,
        )

        logger.info(
            "Parent run_id: %s",
            args.retry_of,
        )

        logger.info(
            "Retry attempt: %s",
            args.retry_attempt,
        )

        # ==============================================================
        # EXECUTE STAGES
        # ==============================================================

        for stage in stages:

            (
                exit_code,
                stage_start,
                stage_end,
                output,
            ) = execute_stage(
                stage=stage,
                project_root=project_root,
                logger=logger,
            )

            duration_seconds = (stage_end - stage_start).total_seconds()

            input_rows, output_rows = collect_stage_metrics(
                spark=spark,
                stage_name=stage.name,
            )

            # ----------------------------------------------------------
            # SUCCESS
            # ----------------------------------------------------------

            if exit_code == 0:

                control.record_stage(
                    run_id=run_id,
                    pipeline_name=PIPELINE_NAME,
                    stage_name=stage.name,
                    status="SUCCESS",
                    start_time=stage_start,
                    end_time=stage_end,
                    input_rows=input_rows,
                    output_rows=output_rows,
                    dq_checks=None,
                    dq_failures=None,
                    error_message=None,
                )

                logger.info(
                    "Control-plane stage recorded: "
                    "%s | SUCCESS | "
                    "duration=%.2f sec | "
                    "input_rows=%s | "
                    "output_rows=%s",
                    stage.name,
                    duration_seconds,
                    input_rows,
                    output_rows,
                )

                continue

            # ----------------------------------------------------------
            # FAILURE
            # ----------------------------------------------------------

            error_message = (
                f"Pipeline stage '{stage.name}' " f"failed with exit code " f"{exit_code}"
            )

            if output:

                output_tail = output[-4000:]

                error_message = f"{error_message}\n\n" f"Stage output:\n" f"{output_tail}"

            control.record_stage(
                run_id=run_id,
                pipeline_name=PIPELINE_NAME,
                stage_name=stage.name,
                status="FAILED",
                start_time=stage_start,
                end_time=stage_end,
                input_rows=input_rows,
                output_rows=output_rows,
                dq_checks=None,
                dq_failures=None,
                error_message=error_message,
            )

            control.complete_pipeline_run(
                run_id=run_id,
                pipeline_name=PIPELINE_NAME,
                status="FAILED",
                start_time=control_start_time,
                end_time=utc_now(),
                error_message=error_message,
            )

            logger.error("=" * 80)
            logger.error("MEDALLION PIPELINE FAILED")
            logger.error(
                "Run ID: %s",
                run_id,
            )
            logger.error(
                "Failed stage: %s",
                stage.name,
            )
            logger.error(
                "Exit code: %s",
                exit_code,
            )
            logger.error("=" * 80)

            return exit_code

        # ==============================================================
        # SUCCESSFUL PIPELINE COMPLETION
        # ==============================================================

        pipeline_end_time = utc_now()

        control.complete_pipeline_run(
            run_id=run_id,
            pipeline_name=PIPELINE_NAME,
            status="SUCCESS",
            start_time=control_start_time,
            end_time=pipeline_end_time,
            error_message=None,
        )

        total_duration = (pipeline_end_time - control_start_time).total_seconds()

        logger.info("=" * 80)
        logger.info("MEDALLION PIPELINE " "COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)

        logger.info(
            "Run ID          : %s",
            run_id,
        )

        logger.info(
            "Parent Run ID   : %s",
            args.retry_of,
        )

        logger.info(
            "Retry Attempt   : %s",
            args.retry_attempt,
        )

        logger.info(
            "Stages executed : %s",
            len(stages),
        )

        logger.info(
            "Total duration  : %.2f sec",
            total_duration,
        )

        logger.info(
            "Completed at    : %s",
            pipeline_end_time.isoformat(),
        )

        logger.info("=" * 80)

        return 0

    except Exception as exc:

        error_message = f"Pipeline orchestration failed: " f"{exc}"

        logger.exception(error_message)

        if control is not None and run_id is not None and control_start_time is not None:

            try:

                control.complete_pipeline_run(
                    run_id=run_id,
                    pipeline_name=PIPELINE_NAME,
                    status="FAILED",
                    start_time=control_start_time,
                    end_time=utc_now(),
                    error_message=error_message,
                )

            except Exception:

                logger.exception("Unable to record pipeline " "failure in control plane.")

        return 1

    finally:

        if spark is not None:
            spark.stop()


if __name__ == "__main__":
    raise SystemExit(main())
