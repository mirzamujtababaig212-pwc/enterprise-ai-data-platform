from pyspark.sql import functions as F

from common.pipelines.gold_pipeline import GoldPipeline
from common.factories.validator_factory import ValidatorFactory
from common.transformers.silver_transformer import SilverTransformer
from spark.transformations.silver_to_gold_transformer import SilverToGoldTransformer
from common.pipelines.pipeline_runtime_config import PipelineRuntimeConfig
from common.readers.delta_reader import DeltaReader
from common.validation.noop_validator import NoOpValidator
from common.writers.delta_writer import DeltaWriter


def test_vehicle_pipeline_bronze_to_silver_to_gold(spark, temp_dir):
    """
    Validate the deterministic Bronze -> Silver -> Gold medallion path
    using isolated temporary Delta tables.

    This intentionally does not execute Kafka or persistent developer tables.
    """

    bronze_df = (
        spark.createDataFrame(
            [
                (
                    "V001",
                    "bronze",
                    0,
                    0,
                    "2026-08-12T00:00:00",
                    '{"vehicle_id":"V001"}',
                    "V001",
                    "2026-08-12T00:00:00",
                    17.5,
                    78.2,
                    45.5,
                    1200,
                    72.0,
                    85.0,
                    88.0,
                    3,
                ),
                (
                    "V001",
                    "bronze",
                    0,
                    1,
                    "2026-08-12T00:01:00",
                    '{"vehicle_id":"V001"}',
                    "V001",
                    "2026-08-12T00:01:00",
                    17.6,
                    78.3,
                    47.0,
                    1210,
                    71.5,
                    84.8,
                    88.5,
                    3,
                ),
                (
                    "V001",
                    "bronze",
                    0,
                    2,
                    "2026-08-12T00:02:00",
                    '{"vehicle_id":"V001"}',
                    "V001",
                    "2026-08-12T00:02:00",
                    17.7,
                    78.4,
                    51.8,
                    1220,
                    71.0,
                    84.5,
                    89.0,
                    4,
                ),
                (
                    "V002",
                    "bronze",
                    0,
                    3,
                    "2026-08-12T00:00:00",
                    '{"vehicle_id":"V002"}',
                    "V002",
                    "2026-08-12T00:00:00",
                    17.8,
                    78.5,
                    32.1,
                    1100,
                    68.0,
                    83.0,
                    87.0,
                    3,
                ),
                (
                    "V002",
                    "bronze",
                    0,
                    4,
                    "2026-08-12T00:01:00",
                    '{"vehicle_id":"V002"}',
                    "V002",
                    "2026-08-12T00:01:00",
                    17.9,
                    78.6,
                    36.4,
                    1110,
                    67.5,
                    82.8,
                    87.5,
                    3,
                ),
                (
                    "V003",
                    "bronze",
                    0,
                    5,
                    "2026-08-12T00:00:00",
                    '{"vehicle_id":"V003"}',
                    "V003",
                    "2026-08-12T00:00:00",
                    18.0,
                    78.7,
                    59.8,
                    1300,
                    55.0,
                    81.0,
                    90.0,
                    4,
                ),
                (
                    "V003",
                    "bronze",
                    0,
                    6,
                    "2026-08-12T00:01:00",
                    '{"vehicle_id":"V003"}',
                    "V003",
                    "2026-08-12T00:01:00",
                    18.1,
                    78.8,
                    61.2,
                    1310,
                    54.5,
                    80.8,
                    90.5,
                    4,
                ),
                (
                    "V003",
                    "bronze",
                    0,
                    7,
                    "2026-08-12T00:02:00",
                    '{"vehicle_id":"V003"}',
                    "V003",
                    "2026-08-12T00:02:00",
                    18.2,
                    78.9,
                    63.5,
                    1320,
                    54.0,
                    80.5,
                    91.0,
                    5,
                ),
                (
                    "V004",
                    "bronze",
                    0,
                    8,
                    "2026-08-12T00:00:00",
                    '{"vehicle_id":"V004"}',
                    "V004",
                    "2026-08-12T00:00:00",
                    18.3,
                    79.0,
                    22.4,
                    900,
                    80.0,
                    86.0,
                    85.0,
                    2,
                ),
                (
                    "V004",
                    "bronze",
                    0,
                    9,
                    "2026-08-12T00:01:00",
                    '{"vehicle_id":"V004"}',
                    "V004",
                    "2026-08-12T00:01:00",
                    18.4,
                    79.1,
                    25.6,
                    910,
                    79.5,
                    85.8,
                    85.5,
                    2,
                ),
                (
                    "V004",
                    "bronze",
                    0,
                    10,
                    "2026-08-12T00:02:00",
                    '{"vehicle_id":"V004"}',
                    "V004",
                    "2026-08-12T00:02:00",
                    18.5,
                    79.2,
                    27.1,
                    920,
                    79.0,
                    85.5,
                    86.0,
                    3,
                ),
            ],
            [
                "kafka_key",
                "kafka_topic",
                "kafka_partition",
                "kafka_offset",
                "kafka_timestamp",
                "raw_value",
                "vehicle_id",
                "event_time",
                "latitude",
                "longitude",
                "speed",
                "rpm",
                "fuel_level",
                "battery",
                "engine_temperature",
                "gear",
            ],
        )
        .withColumn(
            "event_time",
            F.to_timestamp("event_time"),
        )
        .withColumn(
            "kafka_timestamp",
            F.to_timestamp("kafka_timestamp"),
        )
        .withColumn(
            "ingestion_time",
            F.to_timestamp("kafka_timestamp"),
        )
    )

    # The fixture is already Bronze-shaped, so exercise the actual Silver
    # transformation and production Silver validator contract.
    silver_df = SilverTransformer.transform(bronze_df)

    assert silver_df.count() == 11

    validator = ValidatorFactory.create(
        {
            "pipeline": {"class": "silver"},
            "validator": {"type": "default"},
        }
    )

    valid_silver, invalid_silver = validator.validate(silver_df)

    assert invalid_silver.count() == 0
    assert valid_silver.count() == 11

    silver_path = f"{temp_dir}/silver/vehicle_events"
    gold_path = f"{temp_dir}/gold/vehicle_metrics"

    silver_table = "silver.test_vehicle_pipeline_events"
    gold_table = "gold.test_vehicle_pipeline_metrics"

    silver_writer = DeltaWriter(
        table=silver_table,
        path=silver_path,
        mode="merge",
        merge_keys=[
            "vehicle_id",
            "event_time",
        ],
    )

    silver_writer.write(valid_silver)

    first_silver = spark.read.format("delta").load(silver_path)

    assert first_silver.count() == 11
    assert (
        first_silver.select(
            "vehicle_id",
            "event_time",
        )
        .distinct()
        .count()
        == 11
    )

    silver_writer.write(valid_silver)

    updated_silver = valid_silver.withColumn(
        "speed",
        F.when(
            (F.col("vehicle_id") == "V001")
            & (F.col("event_time") == F.to_timestamp(F.lit("2026-08-12 00:00:00"))),
            F.lit(99.9),
        ).otherwise(F.col("speed")),
    )

    silver_writer.write(updated_silver)

    second_silver = spark.read.format("delta").load(silver_path)

    assert second_silver.count() == 11
    assert (
        second_silver.select(
            "vehicle_id",
            "event_time",
        )
        .distinct()
        .count()
        == 11
    )

    updated_row = (
        second_silver.filter(
            (F.col("vehicle_id") == "V001")
            & (F.col("event_time") == F.to_timestamp(F.lit("2026-08-12 00:00:00")))
        )
        .select("speed")
        .collect()
    )

    assert len(updated_row) == 1
    assert updated_row[0]["speed"] == 99.9

    gold_pipeline = GoldPipeline(
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
            pipeline_name="vehicle-pipeline-integration-test",
            enable_validation=False,
            enable_metrics=False,
            enable_dlq=False,
        ),
    )

    result = gold_pipeline.run_batch()

    assert result["eligible_silver_rows"] == 11
    assert result["expected_gold_rows"] == 4
    assert result["actual_gold_rows"] == 4
    assert result["distinct_silver_vehicles"] == 4
    assert result["expected_event_count"] == 11
    assert result["actual_event_count"] == 11
    assert result["mismatched_rows"] == 0

    stored_gold = spark.table(gold_table)

    assert stored_gold.count() == 4

    actual = {row["vehicle_id"]: row.asDict() for row in stored_gold.collect()}

    assert actual["V001"]["event_count"] == 3
    assert actual["V001"]["min_speed"] == 47.0
    assert actual["V001"]["max_speed"] == 99.9

    assert actual["V002"]["event_count"] == 2
    assert actual["V002"]["min_speed"] == 32.1
    assert actual["V002"]["max_speed"] == 36.4

    assert actual["V003"]["event_count"] == 3
    assert actual["V003"]["min_speed"] == 59.8
    assert actual["V003"]["max_speed"] == 63.5

    assert actual["V004"]["event_count"] == 3
    assert actual["V004"]["min_speed"] == 22.4
    assert actual["V004"]["max_speed"] == 27.1

    first_gold = [row.asDict() for row in stored_gold.orderBy("vehicle_id").collect()]

    second_result = gold_pipeline.run_batch()

    assert second_result == result

    second_gold = [row.asDict() for row in spark.table(gold_table).orderBy("vehicle_id").collect()]

    assert second_gold == first_gold
