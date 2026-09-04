from __future__ import annotations

import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    current_timestamp,
    to_timestamp,
)


BRONZE_PATH = os.getenv(
    "BRONZE_PATH",
    "/app/data/lake/bronze/vehicle_telemetry",
)

SILVER_PATH = os.getenv(
    "SILVER_PATH",
    "/app/data/lake/silver/vehicle_telemetry",
)


def create_spark() -> SparkSession:
    return (
        SparkSession.builder.appName("vehicle-telemetry-silver")
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

    bronze = spark.read.format("delta").load(BRONZE_PATH)

    silver = (
        bronze.withColumn(
            "event_time",
            to_timestamp(col("event_time")),
        )
        .withColumn(
            "ingestion_time",
            to_timestamp(col("ingestion_time")),
        )
        .filter(col("event_id").isNotNull())
        .filter(col("vehicle_id").isNotNull())
        .filter(col("event_time").isNotNull())
        .filter(col("latitude").between(-90, 90))
        .filter(col("longitude").between(-180, 180))
        .filter(col("speed") >= 0)
        .filter(col("fuel_level").between(0, 100))
        .filter(col("battery").between(0, 100))
        .dropDuplicates(["event_id"])
        .withColumn("processed_at", current_timestamp())
    )

    (
        silver.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(SILVER_PATH)
    )

    spark.stop()


if __name__ == "__main__":
    run()
