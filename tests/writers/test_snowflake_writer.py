from unittest.mock import MagicMock

from common.writers.snowflake_writer import (
    SnowflakeWriter,
)


def test_snowflake_writer_stores_configuration():

    options = {
        "sfURL": "test.snowflakecomputing.com",
        "sfUser": "test_user",
    }

    writer = SnowflakeWriter(
        options=options,
        table="TEST_TABLE",
        mode="append",
    )

    assert writer.options == options
    assert writer.table == "TEST_TABLE"
    assert writer.mode == "append"


def test_snowflake_writer_writes_dataframe():

    dataframe = MagicMock()

    options = {
        "sfURL": "test.snowflakecomputing.com",
    }

    writer = SnowflakeWriter(
        options=options,
        table="TEST_TABLE",
        mode="append",
    )

    writer.write(dataframe)

    dataframe.write.format.assert_called_once_with("snowflake")

    format_result = dataframe.write.format.return_value

    format_result.options.assert_called_once_with(**options)

    options_result = format_result.options.return_value

    options_result.option.assert_called_once_with(
        "dbtable",
        "TEST_TABLE",
    )

    option_result = options_result.option.return_value

    option_result.mode.assert_called_once_with("append")

    mode_result = option_result.mode.return_value

    mode_result.save.assert_called_once()
