import pytest
from pyspark.sql.types import *

from common.transformers.gold_transformer import GoldTransformer

schema = StructType(
    [
        StructField("vehicle_id", StringType(), True),
        StructField("speed", DoubleType(), True),
    ]
)


class TestGoldTransformer:
    def test_gold_aggregation(self, spark):
        transformer = GoldTransformer()
        df = spark.createDataFrame(
            [
                ("V1", 50),
                ("V1", 60),
                ("V1", 70),
            ],
            ["vehicle_id", "speed"],
        )
        result = transformer.transform(df)
        row = result.collect()[0]
        assert row.avg_speed == 60
        assert row.max_speed == 70
        assert row.min_speed == 50
        assert row.events == 3

    def test_create(self):
        transformer = GoldTransformer()
        assert transformer is not None

    def test_transform_returns_dataframe(self, spark):
        transformer = GoldTransformer()
        df = spark.createDataFrame(
            [("V001", 60.0), ("V002", 80.0)], ["vehicle_id", "speed"]
        )
        result = transformer.transform(df)
        assert result is not None

    def test_row_count(self, spark):
        transformer = GoldTransformer()
        df = spark.createDataFrame(
            [("V001", 60.0), ("V002", 80.0)], ["vehicle_id", "speed"]
        )
        result = transformer.transform(df)
        assert result.select("vehicle_id").distinct().count() == result.count()

    def test_schema(self, spark):
        transformer = GoldTransformer()
        df = spark.createDataFrame([("V001", 60.0)], ["vehicle_id", "speed"])
        result = transformer.transform(df)
        assert "vehicle_id" in result.columns
        assert "avg_speed" in result.columns
        assert "max_speed" in result.columns
        assert "min_speed" in result.columns
        assert "events" in result.columns

    def test_empty_dataframe(self, spark):
        transformer = GoldTransformer()
        empty = spark.createDataFrame([], schema)
        result = transformer.transform(empty)
        assert result.count() == 0

    def test_null_values(self, spark):
        transformer = GoldTransformer()
        df = spark.createDataFrame([(None, 1.0)], schema)
        result = transformer.transform(df)
        assert result.count() == 1

    def test_large_dataset(self, spark):
        transformer = GoldTransformer()
        rows = [(f"V{i}", float(i)) for i in range(5000)]
        df = spark.createDataFrame(rows, ["vehicle_id", "speed"])
        result = transformer.transform(df)
        assert result.count() == 5000

    def test_invalid_schema(self, spark):
        transformer = GoldTransformer()
        df = spark.createDataFrame([("V001",)], ["vehicle_id"])
        with pytest.raises(Exception):
            transformer.transform(df).collect()
