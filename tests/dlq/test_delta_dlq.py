from unittest.mock import MagicMock

import pytest

from common.dlq.delta_dlq import DeltaDLQ


def test_write():
    df = MagicMock()
    df.isEmpty.return_value = False
    writer = MagicMock()
    df.write = writer
    writer.format.return_value = writer
    writer.mode.return_value = writer

    dlq = DeltaDLQ("dlq_table")
    dlq.write(df)

    df.isEmpty.assert_called_once_with()
    writer.format.assert_called_once_with("delta")
    writer.mode.assert_called_once_with("append")
    writer.saveAsTable.assert_called_once_with("dlq_table")


def test_create():
    dlq = DeltaDLQ("test_table")
    assert dlq.table == "test_table"


def test_format():
    df = MagicMock()
    df.isEmpty.return_value = False
    writer = MagicMock()
    df.write = writer
    writer.format.return_value = writer
    writer.mode.return_value = writer

    dlq = DeltaDLQ("table")
    dlq.write(df)

    writer.format.assert_called_with("delta")


def test_mode():
    df = MagicMock()
    df.isEmpty.return_value = False
    writer = MagicMock()
    df.write = writer
    writer.format.return_value = writer
    writer.mode.return_value = writer

    dlq = DeltaDLQ("table")
    dlq.write(df)

    writer.mode.assert_called_with("append")


def test_save():
    df = MagicMock()
    df.isEmpty.return_value = False
    writer = MagicMock()
    df.write = writer
    writer.format.return_value = writer
    writer.mode.return_value = writer

    dlq = DeltaDLQ("table")
    dlq.write(df)

    writer.saveAsTable.assert_called_once_with("table")


def test_none_dataframe_is_ignored():
    dlq = DeltaDLQ("dlq_table")
    dlq.write(None)


def test_empty_dataframe_is_ignored(spark):
    dlq = DeltaDLQ("dlq_table")
    df = spark.createDataFrame(
        [],
        "id INT,name STRING",
    )

    dlq.write(df)

    assert not spark.catalog.tableExists("dlq_table")


def test_nonempty_dataframe_is_written(spark):
    dlq = DeltaDLQ("dlq_table")
    df = spark.createDataFrame(
        [(1, "Alice")],
        ["id", "name"],
    )

    try:
        dlq.write(df)

        assert spark.catalog.tableExists("dlq_table")

        persisted = spark.table("dlq_table")
        assert persisted.count() == 1
        assert persisted.columns == ["id", "name"]
    finally:
        spark.sql("DROP TABLE IF EXISTS dlq_table")


def test_invalid_table():
    dlq = DeltaDLQ("does_not_exist")
    df = MagicMock()
    df.isEmpty.return_value = False
    writer = MagicMock()
    df.write = writer
    writer.format.return_value = writer
    writer.mode.return_value = writer
    writer.saveAsTable.side_effect = RuntimeError("table not found")

    with pytest.raises(RuntimeError):
        dlq.write(df)
