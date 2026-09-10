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

    reader_instance = mock_reader.return_value
    reader_instance.read.return_value = "df"

    result = SparkReader.read_postgres(
        spark,
        "employees",
    )

    mock_reader.assert_called_once_with("employees")
    reader_instance.read.assert_called_once_with(spark)

    assert result == "df"


@patch("common.readers.spark_reader.SnowflakeReader")
@patch("common.readers.spark_reader.Settings")
def test_read_snowflake(mock_settings, mock_reader):

    spark = MagicMock()

    options = {
        "sfURL": "test.snowflakecomputing.com",
        "sfUser": "test_user",
    }

    mock_settings.snowflake.options.return_value = options

    reader_instance = mock_reader.return_value
    reader_instance.read.return_value = "df"

    result = SparkReader.read_snowflake(
        spark,
        "vehicles",
    )

    mock_settings.snowflake.options.assert_called_once_with()

    mock_reader.assert_called_once_with(
        options=options,
        table="vehicles",
    )

    reader_instance.read.assert_called_once_with(spark)

    assert result == "df"


@patch("common.readers.spark_reader.FabricReader")
def test_read_fabric(mock_reader):

    spark = MagicMock()

    reader_instance = mock_reader.return_value
    reader_instance.read.return_value = "df"

    result = SparkReader.read_fabric(
        spark,
        "sales",
    )

    mock_reader.assert_called_once_with("sales")
    reader_instance.read.assert_called_once_with(spark)

    assert result == "df"
