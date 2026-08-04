from unittest.mock import Mock

import pytest

from common.config.settings import Settings
from common.writers.postgres_writer import PostgresWriter


def test_postgres_write_batch():
    writer = PostgresWriter(
        url=Settings.postgres.URL,
        table=Settings.postgres.TABLE,
        properties=Settings.postgres.PROPERTIES,
    )
    df = Mock()
    df.write.jdbc = Mock()
    df.write.mode.return_value = df.write
    writer.write_batch(df)
    df.write.mode.assert_called_once_with("append")
    df.write.jdbc.assert_called_once()


def test_postgres_stream():
    writer = PostgresWriter(
        url=Settings.postgres.URL,
        table=Settings.postgres.TABLE,
        properties=Settings.postgres.PROPERTIES,
    )
    df = Mock()
    stream = Mock()
    df.writeStream = stream
    stream.foreachBatch.return_value = stream
    stream.start.return_value = Mock()
    writer.write_stream(df, Mock())
    stream.foreachBatch.assert_called_once()
    stream.start.assert_called_once()


def test_postgres_writer_failure():

    writer = PostgresWriter(
        url=Settings.postgres.URL,
        table=Settings.postgres.TABLE,
        properties=Settings.postgres.PROPERTIES,
    )

    df = Mock()

    df.write.mode.side_effect = RuntimeError("write failed")

    with pytest.raises(RuntimeError):
        writer.write_batch(df)
