from unittest.mock import Mock

import pytest

from common.writers.delta_writer import DeltaWriter


def test_write_batch():
    writer = DeltaWriter(
        table="bronze.orders",
        checkpoint="/tmp/checkpoint"
    )
    df = Mock()
    (
        df.write
          .format
          .return_value
          .mode
          .return_value
          .saveAsTable
    ) = Mock()
    writer.write_batch(df)
    df.write.format.assert_called_once_with("delta")
    df.write.format.return_value.mode.assert_called_once_with(
        "append"
    )
    (
        df.write
          .format
          .return_value
          .mode
          .return_value
          .saveAsTable
          .assert_called_once_with("bronze.orders")
    )

def test_write_stream():
    writer = DeltaWriter(
        table="bronze.orders",
        checkpoint="/tmp/checkpoint"
    )
    df = Mock()
    stream = Mock()
    df.writeStream = stream
    stream.foreachBatch.return_value = stream
    stream.option.return_value = stream
    stream.outputMode.return_value = stream
    stream.start.return_value = Mock()
    writer.write_stream(
        df,
        Mock()
    )
    stream.foreachBatch.assert_called_once()
    stream.option.assert_called_once_with(
        "checkpointLocation",
        "/tmp/checkpoint"
    )
    stream.outputMode.assert_called_once_with(
        "append"
    )
    stream.start.assert_called_once()

def test_delta_writer_failure():
    writer = DeltaWriter(
        table="bronze.orders",
        checkpoint="/tmp/checkpoint"
    )
    df = Mock()
    df.write.format.side_effect = Exception(
        "write failed"
    )
    with pytest.raises(Exception):
        writer.write_batch(df)
