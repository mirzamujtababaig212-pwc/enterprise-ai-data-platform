from unittest.mock import MagicMock

import pytest

from common.dlq.delta_dlq import DeltaDLQ
from common.dlq.noop_dlq import NoOpDLQ


def test_create():
    dlq = NoOpDLQ()
    assert dlq is not None


def test_write(spark):
    dlq = NoOpDLQ()
    df = spark.createDataFrame([(1, "Alice")], ["id", "name"])
    dlq.write(df)


def test_empty_dataframe(spark):
    dlq = NoOpDLQ()
    empty = spark.createDataFrame(
        [],
        "id INT, name STRING",
    )
    dlq.write(empty)


def test_large_dataset(spark):
    dlq = NoOpDLQ()
    rows = [(i, f"name{i}") for i in range(5000)]
    df = spark.createDataFrame(
        rows,
        ["id", "name"],
    )
    dlq.write(df)


def test_delta_write_with_mock():
    df = MagicMock()
    writer = MagicMock()
    df.write = writer
    writer.format.return_value = writer
    writer.mode.return_value = writer
    writer.saveAsTable.return_value = None
    dlq = DeltaDLQ(table="test_dlq")
    dlq.write(df)
    writer.format.assert_called_once_with("delta")
    writer.mode.assert_called_once_with("append")
    writer.saveAsTable.assert_called_once_with("test_dlq")


def test_invalid_path():
    df = MagicMock()
    writer = MagicMock()
    df.write = writer
    writer.format.return_value = writer
    writer.mode.return_value = writer
    writer.saveAsTable.side_effect = RuntimeError("boom")
    dlq = DeltaDLQ(table="test_dlq")
    with pytest.raises(RuntimeError):
        dlq.write(df)
