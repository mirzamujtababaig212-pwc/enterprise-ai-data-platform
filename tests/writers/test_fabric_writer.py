from unittest.mock import Mock

from common.writers.fabric_writer import FabricWriter


def test_fabric_write_stream_configures_before_start():
    writer = FabricWriter(
        table="silver.vehicle",
        checkpoint="/tmp/fabric-checkpoint",
        output_mode="append",
    )

    df = Mock()
    stream = df.writeStream

    stream.foreachBatch.return_value = stream
    stream.outputMode.return_value = stream
    stream.option.return_value = stream
    stream.queryName.return_value = stream
    stream.start.return_value = Mock()

    foreach_batch = Mock()

    result = writer.write_stream(
        df=df,
        foreach_batch=foreach_batch,
        checkpoint="/tmp/test-checkpoint",
        output_mode="update",
        query_name="vehicle-stream",
    )

    assert result is stream.start.return_value

    stream.foreachBatch.assert_called_once()
    stream.outputMode.assert_called_once_with("update")
    stream.option.assert_called_once_with(
        "checkpointLocation",
        "/tmp/test-checkpoint",
    )
    stream.queryName.assert_called_once_with("vehicle-stream")
    stream.start.assert_called_once()


def test_fabric_write_stream_uses_writer_defaults():
    writer = FabricWriter(
        table="silver.vehicle",
        checkpoint="/tmp/fabric-checkpoint",
        output_mode="append",
    )

    df = Mock()
    stream = df.writeStream

    stream.foreachBatch.return_value = stream
    stream.outputMode.return_value = stream
    stream.option.return_value = stream
    stream.start.return_value = Mock()

    writer.write_stream(
        df=df,
        foreach_batch=Mock(),
    )

    stream.outputMode.assert_called_once_with("append")

    stream.option.assert_called_once_with(
        "checkpointLocation",
        "/tmp/fabric-checkpoint",
    )

    stream.start.assert_called_once()
