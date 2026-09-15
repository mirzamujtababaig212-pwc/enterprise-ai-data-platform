from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

from ml.models.vehicle_risk import FEATURE_COLUMNS


class VehicleRiskFeatureBuilder:
    """Build vehicle-risk model features from Silver telemetry."""

    REQUIRED_INPUT_COLUMNS = (
        "vehicle_id",
        "event_time",
        "speed",
        "rpm",
        "fuel_level",
        "battery",
        "engine_temperature",
    )

    @classmethod
    def build(cls, dataframe: DataFrame) -> DataFrame:
        """Aggregate Silver telemetry to one model-feature row per vehicle."""

        if dataframe is None:
            raise ValueError("dataframe cannot be None")

        missing = [
            column for column in cls.REQUIRED_INPUT_COLUMNS if column not in dataframe.columns
        ]
        if missing:
            raise ValueError(f"missing required columns: {missing}")

        eligible = (
            dataframe.filter(
                F.col("vehicle_id").isNotNull()
                & (F.trim(F.col("vehicle_id")) != "")
                & F.col("event_time").isNotNull()
                & F.col("speed").isNotNull()
                & F.col("rpm").isNotNull()
                & F.col("fuel_level").isNotNull()
                & F.col("battery").isNotNull()
                & F.col("engine_temperature").isNotNull()
            )
            .filter(F.col("speed") >= 0)
            .filter(F.col("rpm") >= 0)
            .filter((F.col("fuel_level") >= 0) & (F.col("fuel_level") <= 100))
            .filter(F.col("battery") >= 0)
        )

        result = eligible.groupBy("vehicle_id").agg(
            F.count(F.lit(1)).cast(IntegerType()).alias("event_count"),
            F.avg("speed").cast("double").alias("avg_speed"),
            F.max("speed").cast("double").alias("max_speed"),
            F.coalesce(
                F.stddev_pop("speed"),
                F.lit(0.0),
            )
            .cast("double")
            .alias("speed_stddev"),
            F.avg("rpm").cast("double").alias("avg_rpm"),
            F.max("rpm").cast("double").alias("max_rpm"),
            F.avg("fuel_level").cast("double").alias("avg_fuel_level"),
            F.min("fuel_level").cast("double").alias("min_fuel_level"),
            F.avg("battery").cast("double").alias("avg_battery"),
            F.avg("engine_temperature").cast("double").alias("avg_engine_temperature"),
            F.max("engine_temperature").cast("double").alias("max_engine_temperature"),
        )

        return result.select(
            "vehicle_id",
            *FEATURE_COLUMNS,
        ).orderBy("vehicle_id")
