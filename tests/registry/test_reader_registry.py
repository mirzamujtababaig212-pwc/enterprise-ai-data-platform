from common.readers.csv_reader import CSVReader
from common.readers.delta_reader import DeltaReader
from common.readers.kafka_reader import KafkaReader
from common.readers.parquet_reader import ParquetReader
from common.registry.reader_registry import READER_REGISTRY
from common.readers.fabric_reader import FabricReader
from common.readers.postgres_reader import PostgresReader
from common.readers.snowflake_reader import SnowflakeReader


def test_registry_contains_kafka():
    assert READER_REGISTRY["kafka"] is KafkaReader


def test_registry_contains_parquet():
    assert READER_REGISTRY["parquet"] is ParquetReader


def test_registry_contains_csv():
    assert READER_REGISTRY["csv"] is CSVReader


def test_registry_contains_delta():
    assert READER_REGISTRY["delta"] is DeltaReader


def test_registry_values_are_classes():
    for cls in READER_REGISTRY.values():
        assert callable(cls)


def test_registry_contains_postgres():
    assert READER_REGISTRY["postgres"] is PostgresReader


def test_registry_contains_snowflake():
    assert READER_REGISTRY["snowflake"] is SnowflakeReader


def test_registry_contains_fabric():
    assert READER_REGISTRY["fabric"] is FabricReader
