from unittest.mock import Mock

import pytest

from common.config.settings import Settings
from common.writers.delta_writer import DeltaWriter


def test_write_batch():
    writer = DeltaWriter(
        table=Settings.storage.BRONZE_TABLE,
        checkpoint=Settings.storage.BRONZE_CHECKPOINT,
    )
    df = Mock()
    df.write.format.return_value.mode.return_value.saveAsTable = Mock()
    writer.write_batch(df)
    df.write.format.assert_called_once_with("delta")
    df.write.format.return_value.mode.assert_called_once_with("append")
    save_as_table = df.write.format.return_value.mode.return_value.saveAsTable
    save_as_table.assert_called_once_with(Settings.storage.BRONZE_TABLE)


def test_write_stream():
    writer = DeltaWriter(
        table=Settings.storage.BRONZE_TABLE,
        checkpoint=Settings.storage.BRONZE_CHECKPOINT,
    )
    df = Mock()
    stream = Mock()
    df.writeStream = stream
    stream.foreachBatch.return_value = stream
    stream.option.return_value = stream
    stream.outputMode.return_value = stream
    stream.start.return_value = Mock()
    writer.write_stream(df, Mock())
    stream.foreachBatch.assert_called_once()
    stream.option.assert_called_once_with("checkpointLocation", Settings.storage.BRONZE_CHECKPOINT)
    stream.outputMode.assert_called_once_with("append")
    stream.start.assert_called_once()


def test_delta_writer_failure():
    writer = DeltaWriter(
        table=Settings.storage.BRONZE_TABLE,
        checkpoint=Settings.storage.BRONZE_CHECKPOINT,
    )
    df = Mock()
    df.write.format.side_effect = RuntimeError("write failed")
    with pytest.raises(RuntimeError):
        writer.write_batch(df)
