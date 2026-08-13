from datetime import datetime

from pyspark.sql.types import (
    StructField,
    StructType,
    StringType,
    IntegerType,
    LongType,
    TimestampType,
)

from common.transformers.bronze_transformer import (
    BronzeTransformer,
)

KAFKA_SCHEMA = StructType(
    [
        StructField("key", StringType(), True),
        StructField("value", StringType(), True),
        StructField("topic", StringType(), True),
        StructField("partition", IntegerType(), True),
        StructField("offset", LongType(), True),
        StructField("timestamp", TimestampType(), True),
    ]
)


def create_kafka_df(spark, rows):

    return spark.createDataFrame(
        rows,
        KAFKA_SCHEMA,
    )


def test_create():

    transformer = BronzeTransformer()

    assert transformer is not None


def test_transform_returns_dataframe(spark):

    now = datetime.now()

    df = create_kafka_df(
        spark,
        [
            (
                "V001",
                '{"vehicle_id":"V001","speed":65.5,"rpm":2500}',
                "vehicle-events",
                0,
                1,
                now,
            ),
            (
                "V002",
                '{"vehicle_id":"V002","speed":72.0,"rpm":2800}',
                "vehicle-events",
                0,
                2,
                now,
            ),
        ],
    )

    result = BronzeTransformer.transform(df)

    assert result.count() == 2
    assert "ingestion_time" in result.columns


def test_schema(spark):

    now = datetime.now()

    df = create_kafka_df(
        spark,
        [
            (
                "V001",
                '{"vehicle_id":"V001","speed":65.5,"rpm":2500}',
                "vehicle-events",
                0,
                1,
                now,
            ),
        ],
    )

    result = BronzeTransformer.transform(df)

    assert "vehicle_id" in result.columns
    assert "speed" in result.columns
    assert "rpm" in result.columns
    assert "kafka_key" in result.columns
    assert "kafka_topic" in result.columns
    assert "kafka_partition" in result.columns
    assert "kafka_offset" in result.columns
    assert "kafka_timestamp" in result.columns
    assert "raw_value" in result.columns
    assert "ingestion_time" in result.columns


def test_values(spark):

    now = datetime.now()

    df = create_kafka_df(
        spark,
        [
            (
                "V001",
                '{"vehicle_id":"V001","speed":65.5,"rpm":2500}',
                "vehicle-events",
                0,
                1,
                now,
            ),
        ],
    )

    result = BronzeTransformer.transform(df)

    row = result.collect()[0]

    assert row["vehicle_id"] == "V001"
    assert row["speed"] == 65.5
    assert row["rpm"] == 2500
    assert row["kafka_key"] == "V001"
    assert row["kafka_topic"] == "vehicle-events"
    assert row["kafka_partition"] == 0
    assert row["kafka_offset"] == 1


def test_empty_dataframe(spark):

    empty = spark.createDataFrame(
        [],
        KAFKA_SCHEMA,
    )

    result = BronzeTransformer.transform(empty)

    assert result.count() == 0


def test_null_values(spark):

    now = datetime.now()

    df = create_kafka_df(
        spark,
        [
            (
                "V001",
                '{"vehicle_id":"V001","speed":null,"rpm":2500}',
                "vehicle-events",
                0,
                1,
                now,
            ),
        ],
    )

    result = BronzeTransformer.transform(df)

    assert result.count() == 1

    row = result.collect()[0]

    assert row["vehicle_id"] == "V001"
    assert row["speed"] is None


def test_large_dataset(spark):

    now = datetime.now()

    rows = [
        (
            f"V{i}",
            f'{{"vehicle_id":"V{i}","speed":{i}.0,"rpm":{1000 + i}}}',
            "vehicle-events",
            0,
            i,
            now,
        )
        for i in range(5000)
    ]

    df = create_kafka_df(
        spark,
        rows,
    )

    result = BronzeTransformer.transform(df)

    assert result.count() == 5000


def test_invalid_json(spark):

    now = datetime.now()

    df = create_kafka_df(
        spark,
        [
            (
                "V001",
                '{"vehicle_id":"V001",}',
                "vehicle-events",
                0,
                1,
                now,
            ),
        ],
    )

    result = BronzeTransformer.transform(df)

    assert result.count() == 1

    row = result.collect()[0]

    assert row["vehicle_id"] is None


def test_bronze_transformer(spark):

    now = datetime.now()

    df = create_kafka_df(
        spark,
        [
            (
                "V1",
                '{"vehicle_id":"V1","speed":60}',
                "vehicle-events",
                0,
                1,
                now,
            ),
        ],
    )

    result = BronzeTransformer.transform(df)

    assert "vehicle_id" in result.columns
    assert "ingestion_time" in result.columns
    assert result.count() == 1
