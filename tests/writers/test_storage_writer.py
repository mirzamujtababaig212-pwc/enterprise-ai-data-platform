from unittest.mock import MagicMock, patch

import pytest

from common.writers.storage_writer import StorageWriter


def test_storage_writer_requires_target():
    df = MagicMock()

    with pytest.raises(
        ValueError,
        match="target is required",
    ):
        StorageWriter.write(
            df=df,
            target=None,
            table="vehicle_dlq",
        )


@patch("common.writers.storage_writer.DeltaWriter")
def test_storage_writer_delta_target_is_case_insensitive(
    mock_delta_writer,
):
    df = MagicMock()

    writer = mock_delta_writer.return_value

    StorageWriter.write(
        df=df,
        target="DELTA",
        table="vehicle_dlq",
        mode="append",
    )

    mock_delta_writer.assert_called_once_with(
        table="vehicle_dlq",
        checkpoint=None,
        mode="append",
    )

    writer.write_batch.assert_called_once_with(df)


@patch("common.writers.storage_writer.ParquetWriter")
def test_storage_writer_parquet_target_is_case_insensitive(
    mock_parquet_writer,
):
    df = MagicMock()

    writer = mock_parquet_writer.return_value

    StorageWriter.write(
        df=df,
        target="PARQUET",
        table="/tmp/vehicle",
        mode="overwrite",
    )

    mock_parquet_writer.assert_called_once_with(
        path="/tmp/vehicle",
        mode="overwrite",
    )

    writer.write_batch.assert_called_once_with(df)


def test_storage_writer_rejects_unknown_target():
    df = MagicMock()

    with pytest.raises(
        ValueError,
        match="Unknown storage writer target",
    ):
        StorageWriter.write(
            df=df,
            target="unknown",
            table="vehicle_dlq",
        )


@pytest.mark.parametrize(
    "target",
    ["postgres", "POSTGRES", "snowflake", "SNOWFLAKE"],
)
def test_storage_writer_requires_configuration_for_database_targets(
    target,
):
    df = MagicMock()

    with pytest.raises(
        ValueError,
        match="requires connection configuration",
    ):
        StorageWriter.write(
            df=df,
            target=target,
            table="vehicle",
        )


def test_storage_writer_supported_targets_match_routable_targets():
    assert StorageWriter.SUPPORTED_TARGETS == {
        "delta",
        "fabric",
        "parquet",
        "console",
    }
