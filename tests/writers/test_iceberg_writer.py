from unittest.mock import Mock

import pytest

from common.writers.iceberg_writer import IcebergWriter


def test_iceberg_writer():
    writer = IcebergWriter(
        table="catalog.db.orders"
    )

    df = Mock()

    df.writeTo.return_value.append = Mock()

    writer.write_batch(df)

    df.writeTo.assert_called_once_with(
        "catalog.db.orders"
    )

    df.writeTo.return_value.append.assert_called_once()


def test_iceberg_writer_failure():

    writer = IcebergWriter(
        table="catalog.db.orders"
    )

    df = Mock()

    df.writeTo.side_effect = Exception(
        "write failed"
    )

    with pytest.raises(Exception):
        writer.write_batch(df)
