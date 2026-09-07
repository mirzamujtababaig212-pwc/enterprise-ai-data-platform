from unittest.mock import MagicMock

from common.readers.snowflake_reader import (
    SnowflakeReader,
)


def test_snowflake_reader_stores_configuration():

    options = {
        "sfURL": "test.snowflakecomputing.com",
        "sfUser": "test_user",
    }

    reader = SnowflakeReader(
        options=options,
        table="TEST_TABLE",
    )

    assert reader.options == options
    assert reader.table == "TEST_TABLE"


def test_snowflake_reader_reads_table():

    spark = MagicMock()

    dataframe = MagicMock()

    spark.read.format.return_value.options.return_value.option.return_value.load.return_value = (
        dataframe
    )

    options = {
        "sfURL": "test.snowflakecomputing.com",
    }

    reader = SnowflakeReader(
        options=options,
        table="TEST_TABLE",
    )

    result = reader.read(spark)

    spark.read.format.assert_called_once_with("snowflake")

    result_options = spark.read.format.return_value.options

    result_options.assert_called_once_with(**options)

    assert result == dataframe
