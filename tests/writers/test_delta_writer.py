from pathlib import Path
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


@pytest.mark.parametrize(
    "cloud_path",
    [
        "s3://bucket/delta/bronze",
        "abfss://container@account.dfs.core.windows.net/delta/bronze",
        "dbfs:/mnt/delta/bronze",
    ],
)
def test_cloud_paths_are_preserved(cloud_path):
    writer = DeltaWriter(
        table="bronze",
        path=cloud_path,
        checkpoint=cloud_path + "/_checkpoint",
    )

    assert writer.path == cloud_path
    assert writer.checkpoint == cloud_path + "/_checkpoint"


def test_local_paths_are_resolved():
    writer = DeltaWriter(
        table="bronze",
        path="./tmp/delta",
        checkpoint="./tmp/checkpoint",
    )

    assert writer.path == Path("./tmp/delta").resolve()
    assert writer.checkpoint == Path("./tmp/checkpoint").resolve()


@patch("common.writers.delta_writer.Path.mkdir")
def test_cloud_stream_paths_do_not_create_local_directories(mock_mkdir):
    cloud_path = "s3://bucket/delta/bronze"
    checkpoint = "s3://bucket/checkpoints/bronze"

    writer = DeltaWriter(
        table="bronze",
        path=cloud_path,
        checkpoint=checkpoint,
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
    )

    mock_mkdir.assert_not_called()

    stream.option.assert_called_once_with(
        "checkpointLocation",
        checkpoint,
    )


def test_register_table_skips_existing_table_when_location_matches():
    writer = DeltaWriter(
        table="bronze.vehicle_events",
        path="s3a://enterprise-data-ai-platform/bronze/vehicle_events",
    )

    spark = Mock()

    spark.catalog.tableExists.return_value = True
    spark.sql.return_value.collect.return_value = [
        Mock(
            col_name="Location", data_type="s3a://enterprise-data-ai-platform/bronze/vehicle_events"
        )
    ]

    writer._register_table(spark)

    spark.catalog.tableExists.assert_called_once_with("bronze.vehicle_events")
    spark.sql.assert_called_once_with("DESCRIBE EXTENDED bronze.vehicle_events")


def test_register_table_matches_local_path_with_spark_file_uri(tmp_path):
    writer = DeltaWriter(
        table="silver.vehicle_events",
        path=str(tmp_path / "silver" / "vehicle_events"),
    )

    spark = Mock()

    spark.catalog.tableExists.return_value = True
    spark.sql.return_value.collect.return_value = [
        Mock(
            col_name="Location",
            data_type=f"file:{tmp_path}/silver/vehicle_events",
        )
    ]

    writer._register_table(spark)

    spark.catalog.tableExists.assert_called_once_with("silver.vehicle_events")
    spark.sql.assert_called_once_with("DESCRIBE EXTENDED silver.vehicle_events")


def test_register_table_raises_on_location_mismatch():
    writer = DeltaWriter(
        table="bronze.vehicle_events",
        path="s3a://enterprise-data-ai-platform/bronze/vehicle_events",
    )

    spark = Mock()

    spark.catalog.tableExists.return_value = True
    spark.sql.return_value.collect.return_value = [
        Mock(
            col_name="Location",
            data_type="file:/home/annie/enterprise_ai_platform/data/delta/bronze/vehicle_events",
        )
    ]

    with pytest.raises(
        RuntimeError,
        match="Delta catalog location mismatch",
    ):
        writer._register_table(spark)

    spark.catalog.tableExists.assert_called_once_with("bronze.vehicle_events")


def test_register_table_creates_missing_table():
    writer = DeltaWriter(
        table="bronze.vehicle_events",
        path="s3a://enterprise-data-ai-platform/bronze/vehicle_events",
    )

    spark = Mock()

    spark.catalog.tableExists.return_value = False

    writer._register_table(spark)

    assert spark.sql.call_count == 2

    calls = [call.args[0] for call in spark.sql.call_args_list]

    assert "CREATE DATABASE IF NOT EXISTS" in calls[0]
    assert "CREATE TABLE bronze.vehicle_events" in calls[1]
    assert "LOCATION 's3a://enterprise-data-ai-platform/bronze/vehicle_events'" in calls[1]


def test_merge_requires_keys():
    with pytest.raises(
        ValueError,
        match="requires at least one merge key",
    ):
        DeltaWriter(
            table=Settings.storage.SILVER_TABLE,
            path=Settings.storage.SILVER_PATH,
            mode="merge",
        )


def test_merge_keys_only_valid_for_merge():
    with pytest.raises(
        ValueError,
        match="only valid with merge mode",
    ):
        DeltaWriter(
            table=Settings.storage.SILVER_TABLE,
            path=Settings.storage.SILVER_PATH,
            mode="append",
            merge_keys=["vehicle_id"],
        )


def test_merge_keys_must_be_unique():
    with pytest.raises(
        ValueError,
        match="must be unique",
    ):
        DeltaWriter(
            table=Settings.storage.SILVER_TABLE,
            path=Settings.storage.SILVER_PATH,
            mode="merge",
            merge_keys=[
                "vehicle_id",
                "vehicle_id",
            ],
        )


def test_merge_keys_are_normalized():
    writer = DeltaWriter(
        table=Settings.storage.SILVER_TABLE,
        path=Settings.storage.SILVER_PATH,
        mode="merge",
        merge_keys=[
            " vehicle_id ",
            " event_time ",
        ],
    )

    assert writer.merge_keys == [
        "vehicle_id",
        "event_time",
    ]


def test_merge_mode_is_batch_only():
    writer = DeltaWriter(
        table=Settings.storage.SILVER_TABLE,
        path=Settings.storage.SILVER_PATH,
        mode="merge",
        merge_keys=["vehicle_id"],
    )

    with pytest.raises(
        ValueError,
        match="only supported for batch writes",
    ):
        writer.write_stream(
            Mock(),
            Mock(),
            checkpoint="/tmp/checkpoint",
        )


@patch("common.writers.delta_writer.DeltaTable")
def test_merge_batch_existing_target(mock_delta_table):
    writer = DeltaWriter(
        table=Settings.storage.SILVER_TABLE,
        path=Settings.storage.SILVER_PATH,
        mode="merge",
        merge_keys=[
            "vehicle_id",
            "event_time",
        ],
    )

    df = Mock()
    df.sparkSession = Mock()

    mock_delta_table.isDeltaTable.return_value = True

    target = mock_delta_table.forPath.return_value
    target_alias = target.alias.return_value
    merge = target_alias.merge.return_value
    matched = merge.whenMatchedUpdateAll.return_value
    not_matched = matched.whenNotMatchedInsertAll.return_value

    with patch.object(
        writer,
        "_register_table",
    ) as mock_register:
        DeltaWriter.write_batch(writer, df)

    mock_delta_table.isDeltaTable.assert_called_once_with(
        df.sparkSession,
        str(writer.path),
    )

    mock_delta_table.forPath.assert_called_once_with(
        df.sparkSession,
        str(writer.path),
    )

    target.alias.assert_called_once_with("target")

    target_alias.merge.assert_called_once_with(
        df.alias.return_value,
        "target.`vehicle_id` = source.`vehicle_id` "
        "AND target.`event_time` = source.`event_time`",
    )

    merge.whenMatchedUpdateAll.assert_called_once()
    matched.whenNotMatchedInsertAll.assert_called_once()
    not_matched.execute.assert_called_once()

    mock_register.assert_called_once_with(df.sparkSession)


@patch("common.writers.delta_writer.DeltaTable")
def test_merge_batch_initializes_missing_target(mock_delta_table):
    writer = DeltaWriter(
        table=Settings.storage.SILVER_TABLE,
        path=Settings.storage.SILVER_PATH,
        mode="merge",
        merge_keys=[
            "vehicle_id",
            "event_time",
        ],
    )

    df = Mock()
    df.sparkSession = Mock()

    mock_delta_table.isDeltaTable.return_value = False

    writer_chain = df.write.format.return_value
    writer_chain.mode.return_value = writer_chain
    writer_chain.option.return_value = writer_chain

    with patch.object(
        writer,
        "_register_table",
    ) as mock_register:
        writer.write_batch(df)

    mock_delta_table.isDeltaTable.assert_called_once_with(
        df.sparkSession,
        str(writer.path),
    )

    df.write.format.assert_called_once_with("delta")
    writer_chain.mode.assert_called_once_with("append")
    writer_chain.option.assert_called_once_with(
        "overwriteSchema",
        "true",
    )
    writer_chain.save.assert_called_once_with(
        str(writer.path),
    )

    mock_delta_table.forPath.assert_not_called()
    mock_register.assert_called_once_with(df.sparkSession)
