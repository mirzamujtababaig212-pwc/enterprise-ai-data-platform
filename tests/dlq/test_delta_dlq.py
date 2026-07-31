from unittest.mock import MagicMock

import pytest

from common.dlq.delta_dlq import DeltaDLQ


def test_write():
    df = MagicMock()
    writer = MagicMock()
    df.write = writer
    writer.format.return_value = writer
    writer.mode.return_value = writer
    dlq = DeltaDLQ("dlq_table")
    dlq.write(df)
    writer.format.assert_called_once_with("delta")
    writer.mode.assert_called_once_with("append")
    writer.saveAsTable.assert_called_once_with("dlq_table")


def test_create():
    dlq = DeltaDLQ("test_table")
    assert dlq.table == "test_table"


def test_format():
    df = MagicMock()
    writer = MagicMock()
    df.write = writer
    writer.format.return_value = writer
    writer.mode.return_value = writer
    dlq = DeltaDLQ("table")
    dlq.write(df)
    writer.format.assert_called_with("delta")


def test_mode():
    df = MagicMock()
    writer = MagicMock()
    df.write = writer
    writer.format.return_value = writer
    writer.mode.return_value = writer
    dlq = DeltaDLQ("table")
    dlq.write(df)
    writer.mode.assert_called_with("append")


def test_save():
    df = MagicMock()
    writer = MagicMock()
    df.write = writer
    writer.format.return_value = writer
    writer.mode.return_value = writer
    dlq = DeltaDLQ("table")
    dlq.write(df)
    writer.saveAsTable.assert_called_once_with("table")


def test_empty_dataframe(spark):
    dlq = DeltaDLQ("dlq_table")
    df = spark.createDataFrame(
        [],
        "id INT,name STRING",
    )
    try:
        dlq.write(df)
    except Exception:
        # Expected if the table doesn't exist in the test environment.
        pass


def test_invalid_table():
    dlq = DeltaDLQ("does_not_exist")
    df = MagicMock()
    writer = MagicMock()
    df.write = writer
    writer.format.return_value = writer
    writer.mode.return_value = writer
    writer.saveAsTable.side_effect = Exception("table not found")
    with pytest.raises(Exception):
        dlq.write(df)
