from unittest.mock import MagicMock

from common.writers.iceberg_writer import IcebergWriter
from common.writers.parquet_writer import ParquetWriter
from common.writers.postgres_writer import PostgresWriter
from common.writers.s3_writer import S3Writer
from common.writers.snowflake_writer import SnowflakeWriter


def _stream_df():
    df = MagicMock()
    df.writeStream = MagicMock()
    writer = df.writeStream
    writer.format.return_value = writer
    writer.foreachBatch.return_value = writer
    writer.outputMode.return_value = writer
    writer.option.return_value = writer
    writer.queryName.return_value = writer
    writer.trigger.return_value = writer
    writer.start.return_value = "query"
    return df, writer


def test_parquet_writer_uses_pipeline_callback():
    df, writer = _stream_df()
    callback = object()

    result = ParquetWriter("/tmp/output").write_stream(
        df=df,
        foreach_batch=callback,
        checkpoint="/tmp/checkpoint",
        output_mode="append",
        query_name="parquet-stream",
        trigger={"processingTime": "10 seconds"},
    )

    writer.foreachBatch.assert_called_once_with(callback)
    writer.option.assert_called_once_with(
        "checkpointLocation",
        "/tmp/checkpoint",
    )
    writer.outputMode.assert_called_once_with("append")
    writer.queryName.assert_called_once_with("parquet-stream")
    writer.trigger.assert_called_once_with(
        processingTime="10 seconds",
    )
    writer.start.assert_called_once_with()
    assert result == "query"


def test_snowflake_writer_uses_pipeline_callback():
    df, writer = _stream_df()
    callback = object()

    result = SnowflakeWriter(
        options={},
        table="vehicle",
    ).write_stream(
        df=df,
        foreach_batch=callback,
        checkpoint="/tmp/checkpoint",
        output_mode="append",
        query_name="snowflake-stream",
        trigger={"once": True},
    )

    writer.foreachBatch.assert_called_once_with(callback)
    writer.option.assert_called_once_with(
        "checkpointLocation",
        "/tmp/checkpoint",
    )
    writer.outputMode.assert_called_once_with("append")
    writer.queryName.assert_called_once_with("snowflake-stream")
    writer.trigger.assert_called_once_with(
        once=True,
    )
    writer.start.assert_called_once_with()
    assert result == "query"


def test_s3_writer_uses_pipeline_callback():
    df, writer = _stream_df()
    callback = object()

    result = S3Writer(
        "/tmp/output",
    ).write_stream(
        df=df,
        foreach_batch=callback,
        checkpoint="/tmp/checkpoint",
        output_mode="append",
        query_name="s3-stream",
        trigger={"availableNow": True},
    )

    writer.foreachBatch.assert_called_once_with(callback)
    writer.option.assert_called_once_with(
        "checkpointLocation",
        "/tmp/checkpoint",
    )
    writer.outputMode.assert_called_once_with("append")
    writer.queryName.assert_called_once_with("s3-stream")
    writer.trigger.assert_called_once_with(
        availableNow=True,
    )
    writer.start.assert_called_once_with()
    assert result == "query"


def test_iceberg_writer_uses_pipeline_callback():
    df, writer = _stream_df()
    callback = object()

    result = IcebergWriter(
        "catalog.db.vehicle",
    ).write_stream(
        df=df,
        foreach_batch=callback,
        checkpoint="/tmp/checkpoint",
        output_mode="append",
        query_name="iceberg-stream",
        trigger={"once": True},
    )

    writer.foreachBatch.assert_called_once_with(callback)
    writer.option.assert_called_once_with(
        "checkpointLocation",
        "/tmp/checkpoint",
    )
    writer.outputMode.assert_called_once_with("append")
    writer.queryName.assert_called_once_with("iceberg-stream")
    writer.trigger.assert_called_once_with(
        once=True,
    )
    writer.start.assert_called_once_with()
    assert result == "query"


def test_postgres_writer_uses_pipeline_callback():
    df, writer = _stream_df()
    callback = object()

    result = PostgresWriter(
        url="jdbc:postgresql://localhost/test",
        table="vehicle",
        properties={},
    ).write_stream(
        df=df,
        foreach_batch=callback,
        checkpoint="/tmp/checkpoint",
        output_mode="append",
        query_name="postgres-stream",
        trigger={"processingTime": "5 seconds"},
    )

    writer.foreachBatch.assert_called_once_with(callback)
    writer.option.assert_called_once_with(
        "checkpointLocation",
        "/tmp/checkpoint",
    )
    writer.outputMode.assert_called_once_with("append")
    writer.queryName.assert_called_once_with("postgres-stream")
    writer.trigger.assert_called_once_with(
        processingTime="5 seconds",
    )
    writer.start.assert_called_once_with()
    assert result == "query"


def test_fabric_writer_uses_pipeline_callback():
    from common.writers.fabric_writer import FabricWriter

    df, writer = _stream_df()
    callback = object()

    result = FabricWriter(
        table="vehicle",
        checkpoint="/tmp/default-checkpoint",
    ).write_stream(
        df=df,
        foreach_batch=callback,
        checkpoint="/tmp/checkpoint",
        output_mode="append",
        query_name="fabric-stream",
        trigger={"processingTime": "10 seconds"},
    )

    writer.foreachBatch.assert_called_once_with(callback)
    writer.option.assert_called_once_with(
        "checkpointLocation",
        "/tmp/checkpoint",
    )
    writer.outputMode.assert_called_once_with("append")
    writer.queryName.assert_called_once_with("fabric-stream")
    writer.trigger.assert_called_once_with(
        processingTime="10 seconds",
    )
    writer.start.assert_called_once_with()
    assert result == "query"


def test_console_writer_uses_native_console_sink():
    from common.writers.console_writer import ConsoleWriter

    df, writer = _stream_df()

    result = ConsoleWriter().write_stream(
        df=df,
        foreach_batch=object(),
        checkpoint="/tmp/checkpoint",
        output_mode=None,
        query_name="console-stream",
        trigger={"once": True},
    )

    writer.outputMode.assert_called_once_with("append")
    writer.option.assert_any_call(
        "checkpointLocation",
        "/tmp/checkpoint",
    )
    writer.queryName.assert_called_once_with("console-stream")
    writer.trigger.assert_called_once_with(
        once=True,
    )
    writer.start.assert_called_once_with()
    assert result == "query"
