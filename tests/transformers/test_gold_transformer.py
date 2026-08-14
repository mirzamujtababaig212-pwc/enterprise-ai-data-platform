import pytest
from pyspark.sql.types import (
    DoubleType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from spark.transformations.silver_to_gold_transformer import (
    SilverToGoldTransformer,
)

schema = StructType(
    [
        StructField(
            "vehicle_id",
            StringType(),
            True,
        ),
        StructField(
            "event_time",
            TimestampType(),
            True,
        ),
        StructField(
            "speed",
            DoubleType(),
            True,
        ),
    ]
)


class TestSilverToGoldTransformer:

    def test_gold_aggregation(self, spark):

        transformer = SilverToGoldTransformer()

        df = spark.createDataFrame(
            [
                ("V1", "2026-01-01 10:00:00", 50.0),
                ("V1", "2026-01-01 10:05:00", 60.0),
                ("V1", "2026-01-01 10:10:00", 70.0),
            ],
            [
                "vehicle_id",
                "event_time",
                "speed",
            ],
        )

        result = transformer.transform(df)

        row = result.collect()[0]

        assert row.vehicle_id == "V1"
        assert row.event_count == 3
        assert row.avg_speed == 60.0
        assert row.min_speed == 50.0
        assert row.max_speed == 70.0

    def test_create(self):

        transformer = SilverToGoldTransformer()

        assert transformer is not None

    def test_transform_returns_dataframe(self, spark):

        transformer = SilverToGoldTransformer()

        df = spark.createDataFrame(
            [
                ("V001", "2026-01-01 10:00:00", 60.0),
                ("V002", "2026-01-01 10:00:00", 80.0),
            ],
            [
                "vehicle_id",
                "event_time",
                "speed",
            ],
        )

        result = transformer.transform(df)

        assert result is not None

    def test_row_count(self, spark):

        transformer = SilverToGoldTransformer()

        df = spark.createDataFrame(
            [
                ("V001", "2026-01-01 10:00:00", 60.0),
                ("V002", "2026-01-01 10:00:00", 80.0),
            ],
            [
                "vehicle_id",
                "event_time",
                "speed",
            ],
        )

        result = transformer.transform(df)

        assert result.count() == 2

    def test_schema(self, spark):

        transformer = SilverToGoldTransformer()

        df = spark.createDataFrame(
            [
                ("V001", "2026-01-01 10:00:00", 60.0),
            ],
            [
                "vehicle_id",
                "event_time",
                "speed",
            ],
        )

        result = transformer.transform(df)

        expected_columns = [
            "vehicle_id",
            "event_count",
            "avg_speed",
            "min_speed",
            "max_speed",
            "first_event_time",
            "last_event_time",
        ]

        assert result.columns == expected_columns

    def test_empty_dataframe(self, spark):

        transformer = SilverToGoldTransformer()

        empty = spark.createDataFrame(
            [],
            schema,
        )

        result = transformer.transform(empty)

        assert result.count() == 0

        assert result.columns == [
            "vehicle_id",
            "event_count",
            "avg_speed",
            "min_speed",
            "max_speed",
            "first_event_time",
            "last_event_time",
        ]

    def test_null_values(self, spark):

        transformer = SilverToGoldTransformer()

        df = spark.createDataFrame(
            [
                ("V001", None, None),
            ],
            schema,
        )

        result = transformer.transform(df)

        assert result.count() == 0

    def test_large_dataset(self, spark):

        transformer = SilverToGoldTransformer()

        rows = [
            (
                f"V{i}",
                f"2026-01-01 10:{i % 60:02d}:00",
                float(i),
            )
            for i in range(5000)
        ]

        df = spark.createDataFrame(
            rows,
            [
                "vehicle_id",
                "event_time",
                "speed",
            ],
        )

        result = transformer.transform(df)

        assert result.count() == 5000

    def test_invalid_schema(self, spark):

        transformer = SilverToGoldTransformer()

        df = spark.createDataFrame(
            [
                ("V001",),
            ],
            ["vehicle_id"],
        )

        with pytest.raises(
            ValueError,
            match="Silver DataFrame is missing required columns",
        ):
            transformer.transform(df)
