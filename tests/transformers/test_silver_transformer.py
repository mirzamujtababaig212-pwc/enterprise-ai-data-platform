from datetime import datetime

import pytest
from pyspark.sql.types import (
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from common.transformers.silver_transformer import SilverTransformer

schema = StructType(
    [
        StructField("vehicle_id", StringType(), True),
        StructField("status", StringType(), True),
        StructField("event_timestamp", TimestampType(), True),
    ]
)


class TestSilverTransformer:
    def test_business_transformation(self, spark):
        transformer = SilverTransformer()
        df = spark.createDataFrame(
            [
                (" V001 ", "running", datetime(2026, 1, 1, 10, 0, 0)),
                ("V002", "idle", datetime(2026, 1, 1, 10, 5, 0)),
            ],
            ["vehicle_id", "status", "event_timestamp"],
        )
        result = transformer.transform(df)
        assert result.count() == 2
        row = result.collect()[0]
        assert row["vehicle_id"] == "V001"
        assert row["status"] == "RUNNING"

    def test_create(self):
        transformer = SilverTransformer()
        assert transformer is not None

    def test_transform_returns_dataframe(self, spark):
        transformer = SilverTransformer()
        df = spark.createDataFrame(
            [(" V001 ", "running", datetime(2026, 1, 1, 10, 0, 0))],
            ["vehicle_id", "status", "event_timestamp"],
        )
        result = transformer.transform(df)
        assert result is not None

    def test_row_count(self, spark):
        transformer = SilverTransformer()
        df = spark.createDataFrame(
            [
                ("V001", "running", datetime(2026, 1, 1, 10, 0, 0)),
                ("V002", "idle", datetime(2026, 1, 1, 10, 5, 0)),
            ],
            ["vehicle_id", "status", "event_timestamp"],
        )
        result = transformer.transform(df)
        assert result.count() == df.count()

    def test_schema(self, spark):
        transformer = SilverTransformer()
        df = spark.createDataFrame(
            [(" V001 ", "running", datetime(2026, 1, 1, 10, 0, 0))],
            ["vehicle_id", "status", "event_timestamp"],
        )
        result = transformer.transform(df)
        assert "vehicle_id" in result.columns
        assert "status" in result.columns
        assert "event_timestamp" in result.columns

    def test_empty_dataframe(self, spark):
        transformer = SilverTransformer()
        empty = spark.createDataFrame([], schema)
        result = transformer.transform(empty)
        assert result.count() == 0

    def test_null_values(self, spark):
        transformer = SilverTransformer()
        df = spark.createDataFrame([(None, None, None)], schema)
        result = transformer.transform(df)
        assert result.count() == 1

    def test_large_dataset(self, spark):
        transformer = SilverTransformer()
        rows = [(f"V{i}", "running", datetime(2026, 1, 1, 10, 0, 0)) for i in range(5000)]
        df = spark.createDataFrame(rows, schema)
        result = transformer.transform(df)
        assert result.count() == 5000

    def test_invalid_schema(self, spark):
        transformer = SilverTransformer()
        df = spark.createDataFrame([(250,)], ["status"])
        with pytest.raises(RuntimeError):
            transformer.transform(df).collect()

    def test_trim_and_upper(self, spark):
        transformer = SilverTransformer()
        df = spark.createDataFrame(
            [("  abc123 ", "running", "2025-01-01")],
            ["vehicle_id", "status", "event_timestamp"],
        )
        result = transformer.transform(df)
        row = result.collect()[0]
        assert row.vehicle_id == "abc123"
        assert row.status == "RUNNING"
