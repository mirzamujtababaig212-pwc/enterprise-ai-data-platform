from __future__ import annotations

import uuid
from datetime import UTC, datetime

from delta.tables import DeltaTable
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import (
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

PIPELINE_RUN_TABLE = "control.pipeline_run_history"
PIPELINE_STAGE_TABLE = "control.pipeline_stage_history"


PIPELINE_RUN_SCHEMA = StructType(
    [
        StructField("run_id", StringType(), False),
        StructField("pipeline_name", StringType(), False),
        StructField("status", StringType(), False),
        StructField("start_time", TimestampType(), False),
        StructField("end_time", TimestampType(), True),
        StructField("duration_seconds", DoubleType(), True),
        StructField("error_message", StringType(), True),
        StructField("parent_run_id", StringType(), True),
        StructField("retry_attempt", LongType(), True),
    ]
)


PIPELINE_STAGE_SCHEMA = StructType(
    [
        StructField("run_id", StringType(), False),
        StructField("pipeline_name", StringType(), False),
        StructField("stage_name", StringType(), False),
        StructField("status", StringType(), False),
        StructField("start_time", TimestampType(), False),
        StructField("end_time", TimestampType(), True),
        StructField("duration_seconds", DoubleType(), True),
        StructField("input_rows", LongType(), True),
        StructField("output_rows", LongType(), True),
        StructField("dq_checks", LongType(), True),
        StructField("dq_failures", LongType(), True),
        StructField("error_message", StringType(), True),
    ]
)


class PipelineControl:
    """
    Production-grade control-plane persistence.

    Stores:

        control.pipeline_run_history
        control.pipeline_stage_history

    Both tables are Delta-backed.

    Run-level capabilities:

        - run history
        - active-run detection
        - latest-run detection
        - retry correlation
        - success-rate metrics
        - duration metrics
        - failure analysis

    Stage-level capabilities:

        - stage history
        - stage success rate
        - duration metrics
        - row-volume metrics
        - data-quality metrics
        - failure analysis
    """

    def __init__(self, spark: SparkSession) -> None:
        self.spark = spark

    # ==============================================================
    # TIME HELPERS
    # ==============================================================

    @staticmethod
    def _utc_now() -> datetime:
        """
        Return UTC datetime without timezone information.

        Spark TimestampType is persisted as a timezone-naive UTC
        timestamp in this local control-plane implementation.
        """
        return datetime.now(UTC).replace(tzinfo=None)

    @staticmethod
    def _normalize_timestamp(
        value: datetime | None,
    ) -> datetime | None:
        """
        Normalize a datetime to naive UTC.
        """
        if value is None:
            return None

        if value.tzinfo is None:
            return value

        return value.astimezone(UTC).replace(tzinfo=None)

    # ==============================================================
    # INITIALIZATION
    # ==============================================================

    def initialize(self) -> None:
        """
        Initialize the control database and tables.
        """
        self.ensure_tables()

    def ensure_tables(self) -> None:
        """
        Create the control database and Delta tables if needed.
        """

        self.spark.sql("CREATE DATABASE IF NOT EXISTS control")

        if not self.spark.catalog.tableExists(PIPELINE_RUN_TABLE):
            empty_df = self.spark.createDataFrame(
                [],
                PIPELINE_RUN_SCHEMA,
            )

            (empty_df.write.format("delta").mode("overwrite").saveAsTable(PIPELINE_RUN_TABLE))

        if not self.spark.catalog.tableExists(PIPELINE_STAGE_TABLE):
            empty_df = self.spark.createDataFrame(
                [],
                PIPELINE_STAGE_SCHEMA,
            )

            (empty_df.write.format("delta").mode("overwrite").saveAsTable(PIPELINE_STAGE_TABLE))

    # ==============================================================
    # RUN ID
    # ==============================================================

    @staticmethod
    def generate_run_id() -> str:
        """
        Generate a UUID-based pipeline run ID.
        """
        return str(uuid.uuid4())

    # ==============================================================
    # ACTIVE RUN DETECTION
    # ==============================================================

    def get_active_runs(
        self,
        pipeline_name: str,
    ) -> DataFrame:
        """
        Return all currently RUNNING executions.
        """

        return self.spark.sql(
            f"""
            SELECT
                run_id,
                pipeline_name,
                status,
                start_time,
                end_time,
                duration_seconds,
                error_message,
                parent_run_id,
                retry_attempt
            FROM {PIPELINE_RUN_TABLE}
            WHERE pipeline_name = '{pipeline_name}'
              AND status = 'RUNNING'
            ORDER BY start_time DESC
            """
        )

    def has_active_run(
        self,
        pipeline_name: str,
    ) -> bool:
        """
        Return True if the pipeline currently has a RUNNING execution.
        """

        return self.get_active_runs(pipeline_name).limit(1).count() > 0

    # ==============================================================
    # LATEST RUN
    # ==============================================================

    def get_latest_run(
        self,
        pipeline_name: str,
    ) -> DataFrame:
        """
        Return the latest execution for a pipeline.
        """

        return self.spark.sql(
            f"""
            SELECT
                run_id,
                pipeline_name,
                status,
                start_time,
                end_time,
                duration_seconds,
                error_message,
                parent_run_id,
                retry_attempt
            FROM {PIPELINE_RUN_TABLE}
            WHERE pipeline_name = '{pipeline_name}'
            ORDER BY start_time DESC
            LIMIT 1
            """
        )

    # ==============================================================
    # PIPELINE START
    # ==============================================================

    def start_pipeline_run(
        self,
        run_id: str,
        pipeline_name: str,
        parent_run_id: str | None = None,
        retry_attempt: int | None = None,
    ) -> datetime:
        """
        Insert a RUNNING pipeline execution.
        """

        if self.has_active_run(pipeline_name):
            raise RuntimeError("Pipeline already has an active RUNNING execution.")

        start_time = self._utc_now()

        row = [
            (
                run_id,
                pipeline_name,
                "RUNNING",
                start_time,
                None,
                None,
                None,
                parent_run_id,
                retry_attempt,
            )
        ]

        df = self.spark.createDataFrame(
            row,
            PIPELINE_RUN_SCHEMA,
        )

        (df.write.format("delta").mode("append").saveAsTable(PIPELINE_RUN_TABLE))

        return start_time

    # ==============================================================
    # PIPELINE COMPLETION
    # ==============================================================

    def complete_pipeline_run(
        self,
        run_id: str,
        pipeline_name: str,
        start_time: datetime,
        status: str = "SUCCESS",
        end_time: datetime | None = None,
        error_message: str | None = None,
    ) -> None:
        """
        Update a pipeline run with its final status.
        """

        normalized_start = self._normalize_timestamp(start_time)

        normalized_end = self._normalize_timestamp(end_time)

        if normalized_start is None:
            raise ValueError("Pipeline start_time cannot be None")

        if normalized_end is None:
            normalized_end = self._utc_now()

        duration_seconds = (normalized_end - normalized_start).total_seconds()

        update_df = self.spark.createDataFrame(
            [
                (
                    run_id,
                    pipeline_name,
                    status,
                    normalized_start,
                    normalized_end,
                    float(duration_seconds),
                    error_message,
                    None,
                    None,
                )
            ],
            PIPELINE_RUN_SCHEMA,
        )

        target = DeltaTable.forName(
            self.spark,
            PIPELINE_RUN_TABLE,
        )

        (
            target.alias("target")
            .merge(
                update_df.alias("source"),
                "target.run_id = source.run_id",
            )
            .whenMatchedUpdate(
                set={
                    "status": "source.status",
                    "end_time": "source.end_time",
                    "duration_seconds": ("source.duration_seconds"),
                    "error_message": "source.error_message",
                }
            )
            .execute()
        )

    # ==============================================================
    # STAGE RECORDING
    # ==============================================================

    def record_stage(
        self,
        run_id: str,
        pipeline_name: str,
        stage_name: str,
        status: str,
        start_time: datetime,
        end_time: datetime,
        input_rows: int | None = None,
        output_rows: int | None = None,
        dq_checks: int | None = None,
        dq_failures: int | None = None,
        error_message: str | None = None,
    ) -> None:
        """
        Insert or update a stage execution record.

        Natural key:

            run_id + stage_name
        """

        normalized_start = self._normalize_timestamp(start_time)

        normalized_end = self._normalize_timestamp(end_time)

        if normalized_start is None:
            raise ValueError("Stage start_time cannot be None")

        if normalized_end is None:
            raise ValueError("Stage end_time cannot be None")

        duration_seconds = (normalized_end - normalized_start).total_seconds()

        stage_df = self.spark.createDataFrame(
            [
                (
                    run_id,
                    pipeline_name,
                    stage_name,
                    status,
                    normalized_start,
                    normalized_end,
                    float(duration_seconds),
                    input_rows,
                    output_rows,
                    dq_checks,
                    dq_failures,
                    error_message,
                )
            ],
            PIPELINE_STAGE_SCHEMA,
        )

        target = DeltaTable.forName(
            self.spark,
            PIPELINE_STAGE_TABLE,
        )

        (
            target.alias("target")
            .merge(
                stage_df.alias("source"),
                """
                target.run_id = source.run_id
                AND target.stage_name = source.stage_name
                """,
            )
            .whenMatchedUpdateAll()
            .whenNotMatchedInsertAll()
            .execute()
        )

    # ==============================================================
    # HISTORY
    # ==============================================================

    def get_pipeline_history(
        self,
        pipeline_name: str,
        limit: int = 20,
    ) -> DataFrame:
        """
        Return recent pipeline run history.
        """

        return self.spark.sql(
            f"""
            SELECT *
            FROM {PIPELINE_RUN_TABLE}
            WHERE pipeline_name = '{pipeline_name}'
            ORDER BY start_time DESC
            LIMIT {int(limit)}
            """
        )

    def get_stage_history(
        self,
        pipeline_name: str,
        limit: int = 50,
    ) -> DataFrame:
        """
        Return recent stage history.
        """

        return self.spark.sql(
            f"""
            SELECT *
            FROM {PIPELINE_STAGE_TABLE}
            WHERE pipeline_name = '{pipeline_name}'
            ORDER BY start_time DESC
            LIMIT {int(limit)}
            """
        )

    def get_pipeline_stages(
        self,
        run_id: str,
    ) -> DataFrame:
        """
        Return all stages belonging to a run.
        """

        return self.spark.sql(
            f"""
            SELECT *
            FROM {PIPELINE_STAGE_TABLE}
            WHERE run_id = '{run_id}'
            ORDER BY start_time
            """
        )

    # ==============================================================
    # SUCCESS METRICS
    # ==============================================================

    def pipeline_success_rate(
        self,
        pipeline_name: str,
    ) -> float:
        """
        Calculate pipeline success percentage.
        """

        result = self.spark.sql(
            f"""
            SELECT
                COUNT(*) AS total_runs,
                SUM(
                    CASE
                        WHEN status = 'SUCCESS'
                        THEN 1
                        ELSE 0
                    END
                ) AS successful_runs
            FROM {PIPELINE_RUN_TABLE}
            WHERE pipeline_name = '{pipeline_name}'
              AND status <> 'RUNNING'
            """
        ).collect()[0]

        total = result["total_runs"]

        if total == 0:
            return 0.0

        return float(result["successful_runs"]) / float(total) * 100.0

    def stage_success_rates(
        self,
        pipeline_name: str,
    ) -> DataFrame:
        """
        Calculate success percentage for each stage.
        """

        return self.spark.sql(
            f"""
            SELECT
                stage_name,
                COUNT(*) AS total_runs,
                SUM(
                    CASE
                        WHEN status = 'SUCCESS'
                        THEN 1
                        ELSE 0
                    END
                ) AS successful_runs,
                ROUND(
                    100.0 *
                    SUM(
                        CASE
                            WHEN status = 'SUCCESS'
                            THEN 1
                            ELSE 0
                        END
                    ) / COUNT(*),
                    2
                ) AS success_rate_pct
            FROM {PIPELINE_STAGE_TABLE}
            WHERE pipeline_name = '{pipeline_name}'
            GROUP BY stage_name
            ORDER BY stage_name
            """
        )

    # ==============================================================
    # DURATION METRICS
    # ==============================================================

    def average_pipeline_duration(
        self,
        pipeline_name: str,
    ) -> float:
        """
        Return average completed pipeline duration.
        """

        result = self.spark.sql(
            f"""
            SELECT AVG(duration_seconds) AS avg_duration
            FROM {PIPELINE_RUN_TABLE}
            WHERE pipeline_name = '{pipeline_name}'
              AND status <> 'RUNNING'
              AND duration_seconds IS NOT NULL
            """
        ).collect()[0]

        return float(result["avg_duration"] or 0.0)

    def average_stage_durations(
        self,
        pipeline_name: str,
    ) -> DataFrame:
        """
        Return average duration for each stage.
        """

        return self.spark.sql(
            f"""
            SELECT
                stage_name,
                AVG(duration_seconds)
                    AS average_duration_seconds
            FROM {PIPELINE_STAGE_TABLE}
            WHERE pipeline_name = '{pipeline_name}'
              AND duration_seconds IS NOT NULL
            GROUP BY stage_name
            ORDER BY stage_name
            """
        )

    # ==============================================================
    # ROW-VOLUME METRICS
    # ==============================================================

    def row_volume_metrics(
        self,
        pipeline_name: str,
    ) -> DataFrame:
        """
        Return input/output volume metrics by stage.
        """

        return self.spark.sql(
            f"""
            SELECT
                stage_name,
                SUM(input_rows) AS total_input_rows,
                SUM(output_rows) AS total_output_rows,
                AVG(input_rows) AS avg_input_rows,
                AVG(output_rows) AS avg_output_rows
            FROM {PIPELINE_STAGE_TABLE}
            WHERE pipeline_name = '{pipeline_name}'
            GROUP BY stage_name
            ORDER BY stage_name
            """
        )

    # ==============================================================
    # DATA QUALITY METRICS
    # ==============================================================

    def data_quality_metrics(
        self,
        pipeline_name: str,
    ) -> DataFrame:
        """
        Return DQ metrics by stage.
        """

        return self.spark.sql(
            f"""
            SELECT
                stage_name,

                SUM(
                    COALESCE(dq_checks, 0)
                ) AS total_dq_checks,

                SUM(
                    COALESCE(dq_failures, 0)
                ) AS total_dq_failures,

                AVG(
                    COALESCE(dq_checks, 0)
                ) AS avg_dq_checks,

                AVG(
                    COALESCE(dq_failures, 0)
                ) AS avg_dq_failures,

                CASE
                    WHEN SUM(
                        COALESCE(dq_checks, 0)
                    ) = 0
                    THEN 0.0

                    ELSE ROUND(
                        100.0 *
                        SUM(
                            COALESCE(dq_failures, 0)
                        )
                        /
                        SUM(
                            COALESCE(dq_checks, 0)
                        ),
                        2
                    )
                END AS dq_failure_rate_pct

            FROM {PIPELINE_STAGE_TABLE}

            WHERE pipeline_name = '{pipeline_name}'

            GROUP BY stage_name

            ORDER BY stage_name
            """
        )

    # ==============================================================
    # FAILURE ANALYSIS
    # ==============================================================

    def failure_analysis(
        self,
        pipeline_name: str,
        limit: int = 20,
    ) -> DataFrame:
        """
        Return recent failed pipeline runs.
        """

        return self.spark.sql(
            f"""
            SELECT
                run_id,
                pipeline_name,
                status,
                start_time,
                end_time,
                duration_seconds,
                error_message,
                parent_run_id,
                retry_attempt
            FROM {PIPELINE_RUN_TABLE}
            WHERE pipeline_name = '{pipeline_name}'
              AND status = 'FAILED'
            ORDER BY start_time DESC
            LIMIT {int(limit)}
            """
        )

    def stage_failure_analysis(
        self,
        pipeline_name: str,
        limit: int = 50,
    ) -> DataFrame:
        """
        Return recent failed stages.
        """

        return self.spark.sql(
            f"""
            SELECT
                run_id,
                pipeline_name,
                stage_name,
                status,
                start_time,
                end_time,
                duration_seconds,
                input_rows,
                output_rows,
                dq_checks,
                dq_failures,
                error_message
            FROM {PIPELINE_STAGE_TABLE}
            WHERE pipeline_name = '{pipeline_name}'
              AND status = 'FAILED'
            ORDER BY start_time DESC
            LIMIT {int(limit)}
            """
        )

    # ==============================================================
    # RETRY CORRELATION
    # ==============================================================

    def get_retry_history(
        self,
        pipeline_name: str,
        limit: int = 50,
    ) -> DataFrame:
        """
        Return retry executions.
        """

        return self.spark.sql(
            f"""
            SELECT
                run_id,
                pipeline_name,
                status,
                start_time,
                end_time,
                duration_seconds,
                parent_run_id,
                retry_attempt,
                error_message
            FROM {PIPELINE_RUN_TABLE}
            WHERE pipeline_name = '{pipeline_name}'
              AND (
                    parent_run_id IS NOT NULL
                    OR retry_attempt IS NOT NULL
                  )
            ORDER BY start_time DESC
            LIMIT {int(limit)}
            """
        )

    def get_run_correlation(
        self,
        run_id: str,
    ) -> DataFrame:
        """
        Return the parent/retry relationship for a run.
        """

        return self.spark.sql(
            f"""
            SELECT
                run_id,
                pipeline_name,
                status,
                start_time,
                end_time,
                duration_seconds,
                parent_run_id,
                retry_attempt,
                error_message
            FROM {PIPELINE_RUN_TABLE}
            WHERE run_id = '{run_id}'
               OR parent_run_id = '{run_id}'
            ORDER BY start_time
            """
        )
