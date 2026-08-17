from __future__ import annotations

from datetime import UTC, datetime

from common.control.pipeline_control import PipelineControl
from common.spark.spark_builder import SparkSessionBuilder


def main() -> None:
    spark = SparkSessionBuilder.build("PipelineControlDiagnostic")

    try:
        control = PipelineControl(spark)

        # Initialize control-plane tables
        control.initialize()

        # Generate a unique pipeline run ID
        run_id = control.generate_run_id()

        # Start pipeline run
        start_time = control.start_pipeline_run(
            run_id=run_id,
            pipeline_name="enterprise_medallion_pipeline",
        )

        # --------------------------------------------------------------
        # Simulate pipeline execution
        # --------------------------------------------------------------

        stage1_start = datetime.now(UTC)

        stage1_end = datetime.now(UTC)

        control.record_stage(
            run_id=run_id,
            pipeline_name="enterprise_medallion_pipeline",
            stage_name="bronze_to_silver",
            status="SUCCESS",
            start_time=stage1_start,
            end_time=stage1_end,
            input_rows=12,
            output_rows=12,
            dq_checks=7,
            dq_failures=0,
        )

        stage2_start = datetime.now(UTC)

        stage2_end = datetime.now(UTC)

        control.record_stage(
            run_id=run_id,
            pipeline_name="enterprise_medallion_pipeline",
            stage_name="silver_to_gold",
            status="SUCCESS",
            start_time=stage2_start,
            end_time=stage2_end,
            input_rows=12,
            output_rows=4,
            dq_checks=14,
            dq_failures=0,
        )

        # --------------------------------------------------------------
        # Complete pipeline run
        # --------------------------------------------------------------

        control.complete_pipeline_run(
            run_id=run_id,
            pipeline_name="enterprise_medallion_pipeline",
            start_time=start_time,
            status="SUCCESS",
            end_time=datetime.now(UTC),
            error_message=None,
        )

        print()
        print("=" * 80)
        print("PIPELINE CONTROL PLANE")
        print("=" * 80)

        print()
        print("Run ID:")
        print(run_id)

        print()
        print("Pipeline history:")

        (
            spark.table("control.pipeline_run_history")
            .orderBy(
                "start_time",
                ascending=False,
            )
            .show(
                10,
                truncate=False,
            )
        )

        print()
        print("Stage history:")

        (
            spark.table("control.pipeline_stage_history")
            .orderBy(
                "start_time",
                ascending=False,
            )
            .show(
                20,
                truncate=False,
            )
        )

        print()
        print("=" * 80)
        print("PIPELINE CONTROL PLANE TEST PASSED")
        print("=" * 80)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
