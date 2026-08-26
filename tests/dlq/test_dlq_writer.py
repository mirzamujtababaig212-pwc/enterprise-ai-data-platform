from unittest.mock import MagicMock, patch

from common.dlq.dlq_writer import DLQWriter


def test_dlq_writer_delegates_to_canonical_storage_writer():
    df = MagicMock()

    with patch("common.dlq.dlq_writer.StorageWriter.write") as mock_write:
        DLQWriter.write(
            df=df,
            target="delta",
            table="vehicle_dlq",
            mode="append",
        )

        mock_write.assert_called_once_with(
            df=df,
            target="delta",
            table="vehicle_dlq",
            mode="append",
        )


def test_dlq_writer_uses_expected_defaults():
    df = MagicMock()

    with patch("common.dlq.dlq_writer.StorageWriter.write") as mock_write:
        DLQWriter.write(df)

        mock_write.assert_called_once_with(
            df=df,
            target="delta",
            table="vehicle_dlq",
            mode="append",
        )
