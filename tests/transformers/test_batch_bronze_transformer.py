from datetime import datetime

import pytest
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from common.transformers.batch_bronze_transformer import BatchBronzeTransformer


schema = StructType(
    [
        StructField("vehicle_id", StringType(), True),
        StructField("event_time", TimestampType(), True),
        StructField("latitude", DoubleType(), True),
        StructField("longitude", DoubleType(), True),
        StructField("speed", DoubleType(), True),
        StructField("rpm", IntegerType(), True),
        StructField("fuel_level", DoubleType(), True),
        StructField("battery", DoubleType(), True),
        StructField("engine_temperature", DoubleType(), True),
        StructField("gear", IntegerType(), True),
    ]
)


def make_row():
    return (
        "V001",
        datetime(2026, 1, 1, 10, 0, 0),
        17.385,
        78.486,
        45.0,
        2000,
        50.0,
        80.0,
        90.0,
        3,
    )


class TestBatchBronzeTransformer:

    def test_transform_adds_canonical_kafka_metadata(self, spark):
        transformer = BatchBronzeTransformer()

        df = spark.createDataFrame(
            [make_row()],
            schema,
        )

        result = transformer.transform(df)

        assert result.count() == 1

        row = result.collect()[0]

        assert row["vehicle_id"] == "V001"
        assert row["kafka_key"] is None
        assert row["kafka_topic"] == "batch"
        assert row["kafka_partition"] is None
        assert row["kafka_offset"] is None
        assert row["kafka_timestamp"] is None
        assert row["raw_value"] is None
        assert row["ingestion_time"] is not None

    def test_preserves_canonical_telemetry_columns(self, spark):
        transformer = BatchBronzeTransformer()

        df = spark.createDataFrame(
            [make_row()],
            schema,
        )

        result = transformer.transform(df)

        for column in schema.fieldNames():
            assert column in result.columns

    def test_missing_required_column_fails(self, spark):
        transformer = BatchBronzeTransformer()

        df = spark.createDataFrame(
            [
                (
                    "V001",
                    datetime(2026, 1, 1, 10, 0, 0),
                    17.385,
                    78.486,
                    45.0,
                    2000,
                    50.0,
                    80.0,
                    90.0,
                )
            ],
            StructType(schema.fields[:-1]),
        )

        with pytest.raises(RuntimeError, match="gear"):
            transformer.transform(df)
