from unittest.mock import MagicMock

from common.config.settings import Settings
from common.readers.fabric_reader import FabricReader
from common.readers.postgres_reader import PostgresReader
from common.readers.snowflake_reader import SnowflakeReader


def test_snowflake_reader_read_delegates_to_read_table():
    spark = MagicMock()
    reader = SnowflakeReader("vehicles")

    result = reader.read(spark)

    spark.read.format.assert_called_once_with("snowflake")
    spark.read.format.return_value.option.assert_called_once_with(
        "dbtable",
        "vehicles",
    )
    spark.read.format.return_value.option.return_value.load.assert_called_once_with()

    assert result == (spark.read.format.return_value.option.return_value.load.return_value)


def test_fabric_reader_read_delegates_to_read_table():
    spark = MagicMock()
    reader = FabricReader("sales")

    result = reader.read(spark)

    spark.read.format.assert_called_once_with("delta")
    spark.read.format.return_value.load.assert_called_once_with("sales")

    assert result == spark.read.format.return_value.load.return_value


def test_postgres_reader_read_delegates_to_read_table():
    spark = MagicMock()
    reader = PostgresReader("vehicles")

    jdbc_reader = spark.read.format.return_value
    jdbc_reader.option.return_value = jdbc_reader

    result = reader.read(spark)

    spark.read.format.assert_called_once_with("jdbc")

    jdbc_reader.option.assert_any_call(
        "url",
        Settings.postgres.URL,
    )
    jdbc_reader.option.assert_any_call(
        "dbtable",
        "vehicles",
    )
    jdbc_reader.option.assert_any_call(
        "user",
        Settings.postgres.USER,
    )
    jdbc_reader.option.assert_any_call(
        "password",
        Settings.postgres.PASSWORD,
    )
    jdbc_reader.load.assert_called_once_with()

    assert result == jdbc_reader.load.return_value
