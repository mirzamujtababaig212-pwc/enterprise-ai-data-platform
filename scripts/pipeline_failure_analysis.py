from __future__ import annotations

import argparse

from common.control.pipeline_control import PipelineControl
from common.spark.spark_builder import SparkSessionBuilder

DEFAULT_PIPELINE = "enterprise_medallion_pipeline"


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline failure analysis")

    parser.add_argument(
        "--pipeline",
        default=DEFAULT_PIPELINE,
        help="Pipeline name",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum failures to display",
    )

    args = parser.parse_args()

    spark = SparkSessionBuilder.build("PipelineFailureAnalysis")

    try:
        control = PipelineControl(spark)

        control.initialize()

        print()
        print("=" * 100)
        print("PIPELINE FAILURE ANALYSIS")
        print("=" * 100)

        print(f"Pipeline: {args.pipeline}")

        print()
        print("=" * 100)
        print("FAILED PIPELINE RUNS")
        print("=" * 100)

        failures = control.failure_analysis(
            args.pipeline,
            limit=args.limit,
        )

        failures.show(truncate=False)

        print()
        print("=" * 100)
        print("FAILED STAGES")
        print("=" * 100)

        stage_failures = control.stage_failure_analysis(
            args.pipeline,
            limit=args.limit,
        )

        stage_failures.show(truncate=False)

        print()
        print("=" * 100)
        print("FAILURE SUMMARY BY STAGE")
        print("=" * 100)

        spark.sql(
            f"""
            SELECT
                stage_name,
                COUNT(*) AS failure_count
            FROM control.pipeline_stage_history
            WHERE pipeline_name = '{args.pipeline}'
              AND status = 'FAILED'
            GROUP BY stage_name
            ORDER BY failure_count DESC
            """
        ).show(truncate=False)

        print()
        print("=" * 100)
        print("RECENT ERROR SIGNATURES")
        print("=" * 100)

        spark.sql(
            f"""
            SELECT
                stage_name,
                error_message,
                COUNT(*) AS occurrence_count
            FROM control.pipeline_stage_history
            WHERE pipeline_name = '{args.pipeline}'
              AND status = 'FAILED'
              AND error_message IS NOT NULL
            GROUP BY
                stage_name,
                error_message
            ORDER BY occurrence_count DESC
            LIMIT {int(args.limit)}
            """
        ).show(truncate=False)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
