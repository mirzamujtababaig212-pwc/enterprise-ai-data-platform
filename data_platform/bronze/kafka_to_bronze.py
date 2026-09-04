from __future__ import annotations

import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, from_json, col
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)


KAFKA_BOOTSTRAP_SERVERS = os.getenv(
    "KAFKA_BOOTSTRAP_SERVERS",
    "kafka:29092",
)

KAFKA_TOPIC = os.getenv(
    "KAFKA_TOPIC",
    "vehicle-telemetry",
)

BRONZE_PATH = os.getenv(
    "BRONZE_PATH",
    "/app/data/lake/bronze/vehicle_telemetry",
)


SCHEMA = StructType(
    [
        StructField("event_id", StringType(), False),
        StructField("schema_version", StringType(), False),
        StructField("source", StringType(), False),
        StructField("vehicle_id", StringType(), False),
        StructField("event_time", StringType(), False),
        StructField("ingestion_time", StringType(), True),
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


def create_spark() -> SparkSession:
    return (
        SparkSession.builder.appName("vehicle-telemetry-bronze")
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

    raw_stream = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
        .option("subscribe", KAFKA_TOPIC)
        .option("startingOffsets", "earliest")
        .load()
    )

    parsed = (
        raw_stream.select(
            from_json(
                col("value").cast("string"),
                SCHEMA,
            ).alias("data"),
            col("timestamp").alias("kafka_timestamp"),
        )
        .select("data.*", "kafka_timestamp")
        .withColumn("processed_at", current_timestamp())
    )

    query = (
        parsed.writeStream.format("delta")
        .outputMode("append")
        .option("checkpointLocation", f"{BRONZE_PATH}/_checkpoint")
        .start(BRONZE_PATH)
    )

    query.awaitTermination()


if __name__ == "__main__":
    run()
