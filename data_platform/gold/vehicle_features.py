from __future__ import annotations

import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    avg,
    count,
    max,
    min,
    stddev,
    window,
    col,
)


SILVER_PATH = os.getenv(
    "SILVER_PATH",
    "/app/data/lake/silver/vehicle_telemetry",
)

GOLD_PATH = os.getenv(
    "GOLD_PATH",
    "/app/data/lake/gold/vehicle_features",
)


def create_spark() -> SparkSession:
    return (
        SparkSession.builder.appName("vehicle-feature-engineering")
        .config(
            "spark.sql.extensions",
            "io.delta.sql.DeltaSparkSessionExtension",
        )
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .getOrCreate()
    )


def run() -> None:
    spark = create_spark()

    silver = spark.read.format("delta").load(SILVER_PATH)

    features = silver.groupBy(
        "vehicle_id",
        window(
            "event_time",
            "5 minutes",
        ),
    ).agg(
        count("*").alias("event_count"),
        avg("speed").alias("avg_speed"),
        max("speed").alias("max_speed"),
        min("speed").alias("min_speed"),
        stddev("speed").alias("speed_stddev"),
        avg("rpm").alias("avg_rpm"),
        max("rpm").alias("max_rpm"),
        avg("fuel_level").alias("avg_fuel_level"),
        min("fuel_level").alias("min_fuel_level"),
        avg("battery").alias("avg_battery"),
        avg("engine_temperature").alias("avg_engine_temperature"),
        max("engine_temperature").alias("max_engine_temperature"),
    )

    flattened = features.select(
        "vehicle_id",
        col("window.start").alias("window_start"),
        col("window.end").alias("window_end"),
        "event_count",
        "avg_speed",
        "max_speed",
        "min_speed",
        "speed_stddev",
        "avg_rpm",
        "max_rpm",
        "avg_fuel_level",
        "min_fuel_level",
        "avg_battery",
        "avg_engine_temperature",
        "max_engine_temperature",
    )

    (
        flattened.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(GOLD_PATH)
    )

    spark.stop()


if __name__ == "__main__":
    run()
