import pytest

from common.factories.writer_factory import WriterFactory
from common.writers.console_writer import ConsoleWriter
from common.writers.delta_writer import DeltaWriter
from common.writers.iceberg_writer import IcebergWriter
from common.writers.postgres_writer import PostgresWriter
from common.writers.s3_writer import S3Writer
from common.config.settings import Settings


def test_create_delta():
    config = {
        "writer": {
            "type": "delta",
            "table": "bronze.vehicle",
            "path": "/tmp/delta/bronze",
            "checkpoint": "/tmp/checkpoint",
        }
    }

    writer = WriterFactory.create(config)

    assert isinstance(writer, DeltaWriter)
    assert writer.table == "bronze.vehicle"


def test_create_postgres():
    config = {
        "writer": {
            "type": "postgres",
            "url": "jdbc:postgresql://localhost/test",
            "table": "vehicle",
            "properties": {},
        }
    }

    writer = WriterFactory.create(config)

    assert isinstance(writer, PostgresWriter)


def test_create_console():
    config = {
        "writer": {
            "type": "console",
        }
    }

    writer = WriterFactory.create(config)

    assert isinstance(writer, ConsoleWriter)


def test_create_s3():
    config = {
        "writer": {
            "type": "s3",
            "path": "/tmp/output",
        }
    }

    writer = WriterFactory.create(config)

    assert isinstance(writer, S3Writer)


def test_create_iceberg():
    config = {
        "writer": {
            "type": "iceberg",
            "table": "catalog.db.vehicle",
        }
    }

    writer = WriterFactory.create(config)

    assert isinstance(writer, IcebergWriter)


def test_invalid_writer():
    config = {
        "writer": {
            "type": "unknown",
        }
    }

    with pytest.raises(ValueError, match="Unknown writer"):
        WriterFactory.create(config)


def test_create_delta_resolves_storage_symbols():
    config = {
        "writer": {
            "type": "delta",
            "table": "BRONZE_TABLE",
            "path": "BRONZE_PATH",
            "checkpoint": "BRONZE_CHECKPOINT",
            "mode": "append",
            "output_mode": "append",
        }
    }

    writer = WriterFactory.create(config)

    assert isinstance(writer, DeltaWriter)
    assert writer.table == Settings.storage.BRONZE_TABLE
    assert str(writer.path) == str(Settings.storage.BRONZE_PATH)
    assert str(writer.checkpoint) == str(Settings.storage.BRONZE_CHECKPOINT)
