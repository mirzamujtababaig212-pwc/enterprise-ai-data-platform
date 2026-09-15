from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from common.aws.glue_table_input import build_delta_table_input


def _bronze_schema():
    return StructType(
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
            StructField("kafka_key", StringType(), True),
            StructField("kafka_topic", StringType(), True),
            StructField("kafka_partition", IntegerType(), True),
            StructField("kafka_offset", LongType(), True),
            StructField("kafka_timestamp", TimestampType(), True),
            StructField("raw_value", StringType(), True),
            StructField("ingestion_time", TimestampType(), True),
        ]
    )


def test_build_delta_table_input():
    table_input = build_delta_table_input(
        table_name="vehicle_events",
        schema=_bronze_schema(),
        location="s3a://enterprise-data-ai-platform/bronze/vehicle_events",
        description="Vehicle event Bronze Delta table.",
    )

    assert table_input["Name"] == "vehicle_events"
    assert table_input["TableType"] == "EXTERNAL_TABLE"
    assert table_input["Description"] == "Vehicle event Bronze Delta table."

    storage_descriptor = table_input["StorageDescriptor"]

    assert storage_descriptor["Location"] == (
        "s3://enterprise-data-ai-platform/bronze/vehicle_events"
    )

    assert storage_descriptor["Columns"] == [
        {"Name": "vehicle_id", "Type": "string"},
        {"Name": "event_time", "Type": "timestamp"},
        {"Name": "latitude", "Type": "double"},
        {"Name": "longitude", "Type": "double"},
        {"Name": "speed", "Type": "double"},
        {"Name": "rpm", "Type": "int"},
        {"Name": "fuel_level", "Type": "double"},
        {"Name": "battery", "Type": "double"},
        {"Name": "engine_temperature", "Type": "double"},
        {"Name": "gear", "Type": "int"},
        {"Name": "kafka_key", "Type": "string"},
        {"Name": "kafka_topic", "Type": "string"},
        {"Name": "kafka_partition", "Type": "int"},
        {"Name": "kafka_offset", "Type": "bigint"},
        {"Name": "kafka_timestamp", "Type": "timestamp"},
        {"Name": "raw_value", "Type": "string"},
        {"Name": "ingestion_time", "Type": "timestamp"},
    ]

    assert table_input["Parameters"] == {
        "classification": "delta",
        "delta.table": "true",
    }


def test_build_delta_table_input_accepts_s3_location():
    table_input = build_delta_table_input(
        table_name="vehicle_events",
        schema=_bronze_schema(),
        location="s3://enterprise-data-ai-platform/bronze/vehicle_events",
    )

    assert (
        table_input["StorageDescriptor"]["Location"]
        == "s3://enterprise-data-ai-platform/bronze/vehicle_events"
    )


def test_build_delta_table_input_rejects_missing_table_name():
    try:
        build_delta_table_input(
            table_name="",
            schema=_bronze_schema(),
            location="s3a://enterprise-data-ai-platform/bronze/vehicle_events",
        )
    except ValueError as exc:
        assert "table name" in str(exc).lower()
    else:
        raise AssertionError("Expected ValueError for missing table name")


def test_build_delta_table_input_rejects_missing_location():
    try:
        build_delta_table_input(
            table_name="vehicle_events",
            schema=_bronze_schema(),
            location="",
        )
    except ValueError as exc:
        assert "location" in str(exc).lower()
    else:
        raise AssertionError("Expected ValueError for missing location")
