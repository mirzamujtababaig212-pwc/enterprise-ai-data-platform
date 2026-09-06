from common.factories.reader_factory import ReaderFactory
from common.readers.csv_reader import CSVReader
from common.readers.delta_reader import DeltaReader
from common.readers.kafka_reader import KafkaReader
from common.readers.parquet_reader import ParquetReader
from common.registry.reader_registry import READER_REGISTRY


def test_create_kafka():
    config = {"reader": {"type": "kafka", "options": {}}}
    reader = ReaderFactory.create(config)
    assert isinstance(reader, KafkaReader)


def test_create_parquet():
    config = {"reader": {"type": "parquet", "path": "/tmp/data"}}
    reader = ReaderFactory.create(config)
    assert isinstance(reader, ParquetReader)


def test_create_csv():
    config = {"reader": {"type": "csv", "path": "/tmp/test.csv"}}
    reader = ReaderFactory.create(config)
    assert isinstance(reader, CSVReader)


def test_create_delta():
    config = {"reader": {"type": "delta", "path": "/tmp/delta"}}
    reader = ReaderFactory.create(config)
    assert isinstance(reader, DeltaReader)


def test_invalid_reader():
    config = {"reader": {"type": "unknown"}}
    import pytest

    with pytest.raises(ValueError):
        ReaderFactory.create(config)


def test_factory_uses_canonical_reader_registry():
    import common.factories.reader_factory as reader_factory

    assert reader_factory.READER_REGISTRY is READER_REGISTRY
