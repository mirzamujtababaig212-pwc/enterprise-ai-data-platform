from unittest.mock import Mock, patch

import pytest

from common.config.settings import Settings
from common.writers.delta_writer import DeltaWriter


@patch("common.writers.delta_writer.Path.mkdir")
def test_write_batch(mock_mkdir):
    writer = DeltaWriter(
        table=Settings.storage.BRONZE_TABLE,
        path=Settings.storage.BRONZE_PATH,
        checkpoint=Settings.storage.BRONZE_CHECKPOINT,
    )

    df = Mock()

    writer_chain = df.write.format.return_value
    writer_chain.mode.return_value = writer_chain
    writer_chain.option.return_value = writer_chain

    with patch.object(
        writer,
        "_register_table",
    ) as mock_register:

        writer.write_batch(df)

    df.write.format.assert_called_once_with("delta")

    writer_chain.mode.assert_called_once_with("append")

    writer_chain.option.assert_called_once_with(
        "overwriteSchema",
        "true",
    )

    writer_chain.save.assert_called_once_with(str(writer.path))

    mock_register.assert_called_once_with(df.sparkSession)


def test_write_aliases_to_batch():
    writer = DeltaWriter(
        table=Settings.storage.BRONZE_TABLE,
        path=Settings.storage.BRONZE_PATH,
        checkpoint=Settings.storage.BRONZE_CHECKPOINT,
    )

    with patch.object(
        writer,
        "write_batch",
    ) as mock_write_batch:

        df = Mock()

        writer.write(df)

        mock_write_batch.assert_called_once_with(df)


@patch("common.writers.delta_writer.Path.mkdir")
def test_write_stream(mock_mkdir):
    writer = DeltaWriter(
        table=Settings.storage.BRONZE_TABLE,
        path=Settings.storage.BRONZE_PATH,
        checkpoint=Settings.storage.BRONZE_CHECKPOINT,
    )

    df = Mock()

    stream = df.writeStream

    stream.outputMode.return_value = stream
    stream.option.return_value = stream
    stream.foreachBatch.return_value = stream

    query = Mock()
    stream.start.return_value = query

    foreach_batch = Mock()

    result = writer.write_stream(
        df,
        foreach_batch,
    )

    assert result is query

    stream.outputMode.assert_called_once_with("append")

    stream.option.assert_called_once_with(
        "checkpointLocation",
        str(writer.checkpoint),
    )

    stream.foreachBatch.assert_called_once_with(foreach_batch)

    stream.start.assert_called_once()


@patch("common.writers.delta_writer.Path.mkdir")
def test_write_stream_explicit_checkpoint(mock_mkdir):
    writer = DeltaWriter(
        table=Settings.storage.BRONZE_TABLE,
        path=Settings.storage.BRONZE_PATH,
    )

    df = Mock()

    stream = df.writeStream

    stream.outputMode.return_value = stream
    stream.option.return_value = stream
    stream.foreachBatch.return_value = stream
    stream.start.return_value = Mock()

    writer.write_stream(
        df,
        Mock(),
        checkpoint="/tmp/test-checkpoint",
    )

    stream.option.assert_called_once_with(
        "checkpointLocation",
        "/tmp/test-checkpoint",
    )


@patch("common.writers.delta_writer.Path.mkdir")
def test_delta_writer_failure(mock_mkdir):
    writer = DeltaWriter(
        table=Settings.storage.BRONZE_TABLE,
        path=Settings.storage.BRONZE_PATH,
        checkpoint=Settings.storage.BRONZE_CHECKPOINT,
    )

    df = Mock()

    df.write.format.side_effect = RuntimeError("write failed")

    with pytest.raises(RuntimeError, match="write failed"):
        writer.write_batch(df)


def test_invalid_mode():
    with pytest.raises(
        ValueError,
        match="Unsupported Delta write mode",
    ):
        DeltaWriter(
            table=Settings.storage.BRONZE_TABLE,
            path=Settings.storage.BRONZE_PATH,
            mode="invalid",
        )


def test_empty_table():
    with pytest.raises(
        ValueError,
        match="table name cannot be empty",
    ):
        DeltaWriter(
            table="",
            path="/tmp/delta",
        )
