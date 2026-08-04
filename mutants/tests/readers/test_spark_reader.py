from unittest.mock import MagicMock, patch

from common.readers.spark_reader import SparkReader


@patch("common.readers.spark_reader.KafkaReader")
def test_read_kafka(mock_kafka):

    spark = MagicMock()

    mock_kafka.read_stream.return_value = "df"

    result = SparkReader.read_kafka(
        spark,
        "vehicle_topic",
        "localhost:9092",
    )

    mock_kafka.read_stream.assert_called_once_with(
        spark,
        "vehicle_topic",
        "localhost:9092",
    )

    assert result == "df"


@patch("common.readers.spark_reader.ParquetReader")
def test_read_parquet(mock_reader):

    spark = MagicMock()

    reader_instance = mock_reader.return_value

    reader_instance.read.return_value = "df"

    result = SparkReader.read_parquet(
        spark,
        "/tmp/data",
        "schema",
    )

    mock_reader.assert_called_once_with(
        "/tmp/data",
        "schema",
    )

    reader_instance.read.assert_called_once_with(spark)

    assert result == "df"


@patch("common.readers.spark_reader.DeltaReader")
def test_read_delta(mock_reader):

    spark = MagicMock()

    reader_instance = mock_reader.return_value

    reader_instance.read.return_value = "df"

    result = SparkReader.read_delta(
        spark,
        "/delta/path",
    )

    mock_reader.assert_called_once_with(
        "/delta/path",
    )

    reader_instance.read.assert_called_once_with(spark)

    assert result == "df"


@patch("common.readers.spark_reader.PostgresReader")
def test_read_postgres(mock_reader):

    spark = MagicMock()

    mock_reader.read_table.return_value = "df"

    result = SparkReader.read_postgres(
        spark,
        "employees",
    )

    mock_reader.read_table.assert_called_once_with(
        spark,
        "employees",
    )

    assert result == "df"


@patch("common.readers.spark_reader.SnowflakeReader")
def test_read_snowflake(mock_reader):

    spark = MagicMock()

    mock_reader.read_table.return_value = "df"

    result = SparkReader.read_snowflake(
        spark,
        "vehicles",
    )

    mock_reader.read_table.assert_called_once_with(
        spark,
        "vehicles",
    )

    assert result == "df"


@patch("common.readers.spark_reader.FabricReader")
def test_read_fabric(mock_reader):

    spark = MagicMock()

    mock_reader.read_table.return_value = "df"

    result = SparkReader.read_fabric(
        spark,
        "sales",
    )

    mock_reader.read_table.assert_called_once_with(
        spark,
        "sales",
    )

    assert result == "df"
