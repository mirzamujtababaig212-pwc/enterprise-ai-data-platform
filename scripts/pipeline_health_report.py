from __future__ import annotations

from common.control.pipeline_control import PipelineControl
from common.spark.spark_builder import SparkSessionBuilder


PIPELINE_NAME = "enterprise_medallion_pipeline"


def print_header(title: str) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def main() -> int:

    print("Initializing Spark...")

    spark = SparkSessionBuilder.build("PipelineHealthReport")

    try:

        control = PipelineControl(spark)

        control.initialize()

        # ====================================================================
        # OPERATIONAL HEALTH
        # ====================================================================

        print_header("OPERATIONAL HEALTH")

        health = control.get_operational_health(PIPELINE_NAME)

        for key, value in health.items():
            print(f"{key:35} : {value}")

        # ====================================================================
        # LATEST RUN
        # ====================================================================

        print_header("LATEST RUN")

        control.get_latest_run(PIPELINE_NAME).show(truncate=False)

        # ====================================================================
        # ACTIVE RUNS
        # ====================================================================

        print_header("ACTIVE RUNS")

        active = control.get_active_runs(PIPELINE_NAME)

        active.show(truncate=False)

        print(
            "Active run count:",
            active.count(),
        )

        # ====================================================================
        # RUN HISTORY
        # ====================================================================

        print_header("RECENT RUN HISTORY")

        control.get_pipeline_history(
            PIPELINE_NAME,
            limit=10,
        ).show(truncate=False)

        # ====================================================================
        # STAGE HISTORY
        # ====================================================================

        print_header("RECENT STAGE HISTORY")

        control.get_stage_history(
            PIPELINE_NAME,
            limit=20,
        ).show(truncate=False)

        # ====================================================================
        # SUCCESS RATE
        # ====================================================================

        print_header("PIPELINE SUCCESS RATE")

        success_rate = control.get_pipeline_success_rate(PIPELINE_NAME)

        print(f"Success rate: {success_rate:.2f}%")

        # ====================================================================
        # STAGE SUCCESS RATES
        # ====================================================================

        print_header("STAGE SUCCESS RATES")

        for stage_name in [
            "bronze_to_silver",
            "silver_to_gold",
        ]:

            rate = control.get_stage_success_rate(
                PIPELINE_NAME,
                stage_name,
            )

            print(f"{stage_name:30} : {rate:.2f}%")

        # ====================================================================
        # AVERAGE PIPELINE DURATION
        # ====================================================================

        print_header("AVERAGE PIPELINE DURATION")

        avg_pipeline_duration = control.get_average_pipeline_duration(PIPELINE_NAME)

        print(
            f"Average duration: " f"{avg_pipeline_duration:.2f} sec"
            if avg_pipeline_duration is not None
            else "Average duration: N/A"
        )

        # ====================================================================
        # AVERAGE STAGE DURATIONS
        # ====================================================================

        print_header("AVERAGE STAGE DURATIONS")

        for stage_name in [
            "bronze_to_silver",
            "silver_to_gold",
        ]:

            duration = control.get_average_stage_duration(
                PIPELINE_NAME,
                stage_name,
            )

            print(
                f"{stage_name:30} : " f"{duration:.2f} sec"
                if duration is not None
                else f"{stage_name:30} : N/A"
            )

        # ====================================================================
        # ROW VOLUME
        # ====================================================================

        print_header("ROW-VOLUME METRICS")

        control.get_row_volume_metrics(PIPELINE_NAME).show(truncate=False)

        # ====================================================================
        # DATA QUALITY
        # ====================================================================

        print_header("DATA-QUALITY METRICS")

        control.get_data_quality_metrics(PIPELINE_NAME).show(truncate=False)

        # ====================================================================
        # FAILURE ANALYSIS
        # ====================================================================

        print_header("RECENT PIPELINE FAILURES")

        failures = control.get_failure_analysis(
            PIPELINE_NAME,
            limit=10,
        )

        failures.show(truncate=False)

        print_header("RECENT FAILED STAGES")

        control.get_failed_stages(
            PIPELINE_NAME,
            limit=10,
        ).show(truncate=False)

        # ====================================================================
        # RETRY HISTORY
        # ====================================================================

        print_header("RETRY HISTORY")

        control.get_retry_history(PIPELINE_NAME).show(truncate=False)

        print(
            "Retry count:",
            control.get_retry_count(PIPELINE_NAME),
        )

        print_header("CONTROL-PLANE HEALTH REPORT COMPLETE")

        return 0

    finally:

        spark.stop()


if __name__ == "__main__":
    raise SystemExit(main())
