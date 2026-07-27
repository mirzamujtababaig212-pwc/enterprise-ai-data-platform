import pytest

from common.builders.writer_builder import WriterBuilder
from common.writers.console_writer import ConsoleWriter
from common.writers.delta_writer import DeltaWriter
from common.writers.iceberg_writer import IcebergWriter
from common.writers.postgres_writer import PostgresWriter
from common.writers.s3_writer import S3Writer


def test_build_delta():

    config = {
        "writer": {
            "type": "delta"
        }
    }

    writer = WriterBuilder.build(
        DeltaWriter,
        config
    )

    assert isinstance(writer, DeltaWriter)


def test_build_postgres():

    config = {
        "writer": {
            "type": "postgres"
        }
    }

    writer = WriterBuilder.build(
        PostgresWriter,
        config
    )

    assert isinstance(writer, PostgresWriter)


def test_build_console():

    config = {
        "writer": {
            "type": "console"
        }
    }

    writer = WriterBuilder.build(
        ConsoleWriter,
        config
    )

    assert isinstance(writer, ConsoleWriter)


def test_build_s3():

    config = {
        "writer": {
            "type": "s3"
        }
    }

    writer = WriterBuilder.build(
        S3Writer,
        config
    )

    assert isinstance(writer, S3Writer)


def test_build_iceberg():

    config = {
        "writer": {
            "type": "iceberg"
        }
    }

    writer = WriterBuilder.build(
        IcebergWriter,
        config
    )

    assert isinstance(writer, IcebergWriter)


def test_invalid_writer():

    config = {
        "writer": {
            "type": "dummy"
        }
    }

    with pytest.raises(ValueError):
        WriterBuilder.build(None, config)
