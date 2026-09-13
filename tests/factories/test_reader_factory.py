from common.factories.reader_factory import ReaderFactory
from common.readers.csv_reader import CSVReader
from common.readers.delta_reader import DeltaReader
from common.readers.kafka_reader import KafkaReader
from common.readers.parquet_reader import ParquetReader
from common.readers.fabric_reader import FabricReader
from common.readers.postgres_reader import PostgresReader
from common.readers.snowflake_reader import SnowflakeReader


def test_create_kafka():
    config = {"reader": {"type": "kafka", "options": {}}}

    reader = ReaderFactory.create(config)

    assert isinstance(reader, KafkaReader)


def test_create_parquet():
    config = {
        "reader": {
            "type": "parquet",
            "path": "/tmp/data",
        }
    }

    reader = ReaderFactory.create(config)

    assert isinstance(reader, ParquetReader)
    assert str(reader.path) == "/tmp/data"


def test_create_csv():
    config = {
        "reader": {
            "type": "csv",
            "path": "/tmp/test.csv",
        }
    }

    reader = ReaderFactory.create(config)

    assert isinstance(reader, CSVReader)
    assert str(reader.path) == "/tmp/test.csv"


def test_create_delta():
    config = {
        "reader": {
            "type": "delta",
            "path": "/tmp/delta",
        }
    }

    reader = ReaderFactory.create(config)

    assert isinstance(reader, DeltaReader)
    assert str(reader.path) == "/tmp/delta"


def test_create_postgres():
    config = {
        "reader": {
            "type": "postgres",
            "table": "vehicles",
        }
    }

    reader = ReaderFactory.create(config)

    assert isinstance(reader, PostgresReader)
    assert reader.table == "vehicles"


def test_create_snowflake():
    config = {
        "reader": {
            "type": "snowflake",
            "table": "vehicles",
        }
    }

    reader = ReaderFactory.create(config)

    assert isinstance(reader, SnowflakeReader)
    assert reader.table == "vehicles"


def test_create_fabric():
    config = {
        "reader": {
            "type": "fabric",
            "table": "sales",
        }
    }

    reader = ReaderFactory.create(config)

    assert isinstance(reader, FabricReader)
    assert reader.table == "sales"


def test_invalid_reader():
    config = {
        "reader": {
            "type": "unknown",
        }
    }

    import pytest

    with pytest.raises(ValueError):
        ReaderFactory.create(config)


def test_create_s3():
    from common.readers.s3_reader import S3Reader

    config = {
        "reader": {
            "type": "s3",
            "path": "s3://bucket/bronze/events",
        }
    }

    reader = ReaderFactory.create(config)

    assert isinstance(reader, S3Reader)
    assert reader.path == "s3://bucket/bronze/events"


def test_create_s3_requires_path():
    config = {
        "reader": {
            "type": "s3",
        }
    }

    import pytest

    with pytest.raises(ValueError):
        ReaderFactory.create(config)
