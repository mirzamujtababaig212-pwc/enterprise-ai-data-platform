from unittest.mock import Mock

from common.readers.s3_reader import S3Reader


def test_s3_reader_stores_path_and_schema():
    schema = Mock()

    reader = S3Reader(
        path="s3://bucket/bronze/events",
        schema=schema,
    )

    assert reader.path == "s3://bucket/bronze/events"
    assert reader.schema is schema


def test_s3_reader_reads_parquet():
    reader = S3Reader(
        path="s3://bucket/bronze/events",
    )

    spark = Mock()
    spark.read.parquet.return_value = "dataframe"

    result = reader.read(spark)

    assert result == "dataframe"
    spark.read.parquet.assert_called_once_with(
        "s3://bucket/bronze/events",
    )


def test_s3_reader_applies_schema():
    schema = Mock()

    reader = S3Reader(
        path="s3://bucket/bronze/events",
        schema=schema,
    )

    spark = Mock()
    spark.read.schema.return_value.parquet.return_value = "dataframe"

    result = reader.read(spark)

    assert result == "dataframe"
    spark.read.schema.assert_called_once_with(schema)
    spark.read.schema.return_value.parquet.assert_called_once_with(
        "s3://bucket/bronze/events",
    )


def test_s3_reader_read_delegates_to_read_table():
    reader = S3Reader(
        path="s3://bucket/bronze/events",
    )

    expected = object()

    original = S3Reader.read_table

    try:
        S3Reader.read_table = Mock(return_value=expected)

        spark = Mock()

        result = reader.read(spark)

        assert result is expected
        S3Reader.read_table.assert_called_once_with(
            spark=spark,
            path="s3://bucket/bronze/events",
            schema=None,
        )
    finally:
        S3Reader.read_table = original
