import pytest

from common.builders.reader_builder import ReaderBuilder
from common.readers.csv_reader import CSVReader
from common.readers.delta_reader import DeltaReader
from common.readers.fabric_reader import FabricReader
from common.readers.kafka_reader import KafkaReader
from common.readers.parquet_reader import ParquetReader
from common.readers.postgres_reader import PostgresReader
from common.readers.snowflake_reader import SnowflakeReader


def test_build_kafka():
    config = {"reader": {"type": "kafka"}}

    reader = ReaderBuilder.build(config)

    assert isinstance(reader, KafkaReader)


def test_build_parquet():
    config = {"reader": {"type": "parquet"}}

    reader = ReaderBuilder.build(config)

    assert isinstance(reader, ParquetReader)


def test_build_csv():
    config = {"reader": {"type": "csv"}}

    reader = ReaderBuilder.build(config)

    assert isinstance(reader, CSVReader)


def test_build_delta():
    config = {"reader": {"type": "delta"}}

    reader = ReaderBuilder.build(config)

    assert isinstance(reader, DeltaReader)


def test_build_postgres():
    config = {
        "reader": {
            "type": "postgres",
            "table": "vehicles",
        }
    }

    reader = ReaderBuilder.build(config)

    assert isinstance(reader, PostgresReader)
    assert reader.table == "vehicles"


def test_build_snowflake():
    config = {
        "reader": {
            "type": "snowflake",
            "table": "vehicles",
        }
    }

    reader = ReaderBuilder.build(config)

    assert isinstance(reader, SnowflakeReader)
    assert reader.table == "vehicles"


def test_build_fabric():
    config = {
        "reader": {
            "type": "fabric",
            "table": "sales",
        }
    }

    reader = ReaderBuilder.build(config)

    assert isinstance(reader, FabricReader)
    assert reader.table == "sales"


def test_build_requires_config():
    with pytest.raises(ValueError, match="configuration cannot be empty"):
        ReaderBuilder.build({})


def test_build_requires_reader_type():
    with pytest.raises(ValueError, match="Reader type is required"):
        ReaderBuilder.build({"reader": {}})


def test_build_rejects_unknown_reader():
    config = {
        "reader": {
            "type": "unknown",
        }
    }

    with pytest.raises(ValueError, match="Unsupported reader type"):
        ReaderBuilder.build(config)


def test_build_table_reader_requires_table():
    config = {
        "reader": {
            "type": "postgres",
        }
    }

    with pytest.raises(ValueError, match="Table is required"):
        ReaderBuilder.build(config)
