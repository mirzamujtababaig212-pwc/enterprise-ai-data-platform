from pyspark.sql import functions as F

from common.pipelines.gold_pipeline import GoldPipeline
from common.pipelines.pipeline_runtime_config import PipelineRuntimeConfig
from common.readers.delta_reader import DeltaReader
from common.validation.noop_validator import NoOpValidator
from common.writers.delta_writer import DeltaWriter
from spark.transformations.silver_to_gold_transformer import (
    SilverToGoldTransformer,
)


def test_gold_pipeline_runs_silver_to_persisted_gold(
    spark,
    temp_dir,
):
    silver_path = f"{temp_dir}/silver/vehicle_events"
    gold_path = f"{temp_dir}/gold/vehicle_metrics"

    silver_table = "silver.test_vehicle_events"
    gold_table = "gold.test_vehicle_metrics"

    silver_df = spark.createDataFrame(
        [
            ("V001", "2026-01-01 10:00:00", 45.5),
            ("V001", "2026-01-01 10:05:00", 48.0),
            ("V001", "2026-01-01 10:10:00", 50.01),
            ("V002", "2026-01-01 10:00:00", 32.1),
            ("V002", "2026-01-01 10:05:00", 36.4),
            (None, "2026-01-01 10:10:00", 99.0),
            ("V003", "2026-01-01 10:15:00", -5.0),
        ],
        ["vehicle_id", "event_time", "speed"],
    ).withColumn(
        "event_time",
        F.to_timestamp("event_time"),
    )

    DeltaWriter(
        table=silver_table,
        path=silver_path,
        mode="overwrite",
    ).write(silver_df)

    pipeline = GoldPipeline(
        spark=spark,
        reader=DeltaReader(
            path=silver_path,
            table=silver_table,
        ),
        writer=DeltaWriter(
            table=gold_table,
            path=gold_path,
            mode="overwrite",
        ),
        transformer=SilverToGoldTransformer(),
        validator=NoOpValidator(),
        metrics=None,
        dlq=None,
        config=PipelineRuntimeConfig(
            pipeline_name="gold-test",
            enable_validation=False,
            enable_metrics=False,
            enable_dlq=False,
        ),
    )

    result = pipeline.run_batch()

    assert result["eligible_silver_rows"] == 5
    assert result["expected_gold_rows"] == 2
    assert result["actual_gold_rows"] == 2
    assert result["distinct_silver_vehicles"] == 2
    assert result["expected_event_count"] == 5
    assert result["actual_event_count"] == 5
    assert result["mismatched_rows"] == 0

    stored_gold = spark.table(gold_table)

    assert stored_gold.count() == 2

    rows = {row.vehicle_id: row for row in stored_gold.collect()}

    assert rows["V001"].event_count == 3
    assert rows["V001"].min_speed == 45.5
    assert rows["V001"].max_speed == 50.01

    assert rows["V002"].event_count == 2
    assert rows["V002"].min_speed == 32.1
    assert rows["V002"].max_speed == 36.4
