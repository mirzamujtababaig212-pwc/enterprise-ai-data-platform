from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from common.aws.glue_schema import spark_schema_to_glue_columns


def test_spark_schema_maps_to_glue_columns():
    schema = StructType(
        [
            StructField("vehicle_id", StringType(), True),
            StructField("event_time", TimestampType(), True),
            StructField("latitude", DoubleType(), True),
            StructField("rpm", IntegerType(), True),
            StructField("kafka_offset", LongType(), True),
        ]
    )

    assert spark_schema_to_glue_columns(schema) == [
        {"Name": "vehicle_id", "Type": "string"},
        {"Name": "event_time", "Type": "timestamp"},
        {"Name": "latitude", "Type": "double"},
        {"Name": "rpm", "Type": "int"},
        {"Name": "kafka_offset", "Type": "bigint"},
    ]


def test_spark_schema_preserves_column_order():
    schema = StructType(
        [
            StructField("z_column", StringType(), True),
            StructField("a_column", StringType(), True),
        ]
    )

    columns = spark_schema_to_glue_columns(schema)

    assert [column["Name"] for column in columns] == [
        "z_column",
        "a_column",
    ]


def test_spark_schema_rejects_unsupported_type():
    from pyspark.sql.types import BinaryType

    schema = StructType(
        [
            StructField("payload", BinaryType(), True),
        ]
    )

    try:
        spark_schema_to_glue_columns(schema)
    except ValueError as exc:
        assert "Unsupported Spark data type" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsupported Spark type")
