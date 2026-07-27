import pytest

from common.factories.writer_factory import WriterFactory
from common.writers.console_writer import ConsoleWriter
from common.writers.delta_writer import DeltaWriter
from common.writers.iceberg_writer import IcebergWriter
from common.writers.postgres_writer import PostgresWriter
from common.writers.s3_writer import S3Writer


def test_create_delta():

    config = {
        "writer": {
            "type": "delta",
            "table": "bronze.vehicle",
            "checkpoint": "/tmp/checkpoint"
        }
    }

    writer = WriterFactory.create(config)

    assert isinstance(writer, DeltaWriter)


def test_create_postgres():

    config = {
        "writer": {
            "type": "postgres",
            "url": "jdbc:postgresql://localhost/test",
            "table": "vehicle",
            "properties": {}
        }
    }

    writer = WriterFactory.create(config)

    assert isinstance(writer, PostgresWriter)


def test_create_console():

    config = {
        "writer": {
            "type": "console"
        }
    }

    writer = WriterFactory.create(config)

    assert isinstance(writer, ConsoleWriter)


def test_create_s3():

    config = {
        "writer": {
            "type": "s3",
            "path": "/tmp/output"
        }
    }

    writer = WriterFactory.create(config)

    assert isinstance(writer, S3Writer)


def test_create_iceberg():

    config = {
        "writer": {
            "type": "iceberg",
            "table": "catalog.db.vehicle"
        }
    }

    writer = WriterFactory.create(config)

    assert isinstance(writer, IcebergWriter)


def test_invalid_writer():

    config = {
        "writer": {
            "type": "unknown"
        }
    }

    with pytest.raises(ValueError):
        WriterFactory.create(config)
