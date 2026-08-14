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

from common.transformers.silver_transformer import SilverTransformer

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


def make_row(
    vehicle_id="V001",
    event_time=datetime(2026, 1, 1, 10, 0, 0),
    latitude=17.385,
    longitude=78.486,
    speed=45.0,
    rpm=2000,
    fuel_level=50.0,
    battery=80.0,
    engine_temperature=90.0,
    gear=3,
):
    return (
        vehicle_id,
        event_time,
        latitude,
        longitude,
        speed,
        rpm,
        fuel_level,
        battery,
        engine_temperature,
        gear,
    )


class TestSilverTransformer:
    def make_row(
        vehicle_id="V001",
        event_time=datetime(2026, 1, 1, 10, 0, 0),
        latitude=17.385,
        longitude=78.486,
        speed=45.0,
        rpm=2000,
        fuel_level=50.0,
        battery=80.0,
        engine_temperature=90.0,
        gear=3,
    ):
        return (
            vehicle_id,
            event_time,
            latitude,
            longitude,
            speed,
            rpm,
            fuel_level,
            battery,
            engine_temperature,
            gear,
        )

    def test_business_transformation(self, spark):
        transformer = SilverTransformer()

        df = spark.createDataFrame(
            [
                make_row(
                    vehicle_id=" V001 ",
                    speed=45.0,
                    fuel_level=50.0,
                    battery=80.0,
                    engine_temperature=90.0,
                ),
                make_row(
                    vehicle_id="V002",
                    speed=10.0,
                    fuel_level=10.0,
                    battery=15.0,
                    engine_temperature=120.0,
                ),
            ],
            schema,
        )

        result = transformer.transform(df)

        assert result.count() == 2

        rows = {row["vehicle_id"]: row for row in result.collect()}

        row = rows["V001"]

        assert row["speed_category"] == "NORMAL"
        assert row["fuel_status"] == "NORMAL"
        assert row["battery_status"] == "NORMAL"
        assert row["vehicle_status"] == "NORMAL"

        row = rows["V002"]

        assert row["speed_category"] == "LOW"
        assert row["fuel_status"] == "CRITICAL"
        assert row["battery_status"] == "CRITICAL"
        assert row["vehicle_status"] == "BATTERY_CRITICAL"

    def test_create(self):
        transformer = SilverTransformer()
        assert transformer is not None

    def test_transform_returns_dataframe(self, spark):
        transformer = SilverTransformer()

        df = spark.createDataFrame(
            [make_row()],
            schema,
        )

        result = transformer.transform(df)

        assert result is not None

    def test_row_count(self, spark):
        transformer = SilverTransformer()

        df = spark.createDataFrame(
            [
                make_row(vehicle_id="V001"),
                make_row(vehicle_id="V002"),
            ],
            schema,
        )

        result = transformer.transform(df)

        assert result.count() == df.count()

    def test_schema(self, spark):
        transformer = SilverTransformer()

        df = spark.createDataFrame(
            [make_row()],
            schema,
        )

        result = transformer.transform(df)

        assert "vehicle_id" in result.columns
        assert "event_time" in result.columns
        assert "speed" in result.columns
        assert "fuel_level" in result.columns
        assert "battery" in result.columns
        assert "engine_temperature" in result.columns

        assert "speed_category" in result.columns
        assert "fuel_status" in result.columns
        assert "battery_status" in result.columns
        assert "vehicle_status" in result.columns

    def test_empty_dataframe(self, spark):
        transformer = SilverTransformer()
        empty = spark.createDataFrame([], schema)
        result = transformer.transform(empty)
        assert result.count() == 0

    def test_null_values(self, spark):
        transformer = SilverTransformer()

        df = spark.createDataFrame(
            [
                (
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
            ],
            schema,
        )

        result = transformer.transform(df)

        assert result.count() == 1

    def test_large_dataset(self, spark):
        transformer = SilverTransformer()

        rows = [make_row(vehicle_id=f"V{i}") for i in range(5000)]

        df = spark.createDataFrame(rows, schema)

        result = transformer.transform(df)

        assert result.count() == 5000

    def test_invalid_schema(self, spark):
        transformer = SilverTransformer()

        df = spark.createDataFrame(
            [(250,)],
            ["status"],
        )

        with pytest.raises(RuntimeError):
            transformer.transform(df)

    def test_trim_vehicle_id(self, spark):
        transformer = SilverTransformer()

        df = spark.createDataFrame(
            [
                make_row(
                    vehicle_id="  abc123  ",
                )
            ],
            schema,
        )

        result = transformer.transform(df)
        row = result.collect()[0]
        assert row.vehicle_id == "abc123"

    def test_deduplicates_vehicle_event_time(self, spark):
        transformer = SilverTransformer()

        event_time = datetime(2026, 1, 1, 10, 0, 0)

        df = spark.createDataFrame(
            [
                make_row(
                    vehicle_id="V001",
                    event_time=event_time,
                    speed=40.0,
                ),
                make_row(
                    vehicle_id="V001",
                    event_time=event_time,
                    speed=40.0,
                ),
                make_row(
                    vehicle_id="V002",
                    event_time=event_time,
                    speed=40.0,
                ),
            ],
            schema,
        )

        result = transformer.transform(df)

        assert result.count() == 2
