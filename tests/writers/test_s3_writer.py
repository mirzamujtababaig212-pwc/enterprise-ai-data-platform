from unittest.mock import Mock

import pytest

from common.writers.s3_writer import S3Writer


def test_s3_writer():

    writer = S3Writer(path="s3://bucket/orders")

    df = Mock()

    df.write.mode.return_value.parquet = Mock()

    writer.write_batch(df)

    df.write.mode.assert_called_once_with("append")

    df.write.mode.return_value.parquet.assert_called_once_with("s3://bucket/orders")


def test_s3_writer_failure():

    writer = S3Writer(path="s3://bucket/orders")

    df = Mock()

    df.write.mode.side_effect = Exception("write failed")

    with pytest.raises(Exception):
        writer.write_batch(df)
