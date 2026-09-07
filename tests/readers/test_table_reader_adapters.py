from unittest.mock import MagicMock

from common.config.settings import Settings
from common.readers.fabric_reader import FabricReader
from common.readers.postgres_reader import PostgresReader
from common.readers.snowflake_reader import SnowflakeReader


def test_snowflake_reader_read_delegates_to_read_table(
    monkeypatch,
):

    options = {
        "sfURL": ("test.snowflakecomputing.com"),
    }

    reader = SnowflakeReader(
        options=options,
        table="vehicles",
    )

    expected = object()

    captured = {}

    def fake_read_table(
        spark,
        options,
        table,
    ):
        captured["spark"] = spark
        captured["options"] = options
        captured["table"] = table

        return expected

    monkeypatch.setattr(
        SnowflakeReader,
        "read_table",
        staticmethod(fake_read_table),
    )

    spark = object()

    result = reader.read(spark)

    assert result is expected

    assert captured["spark"] is spark

    assert captured["options"] == options

    assert captured["table"] == "vehicles"


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
