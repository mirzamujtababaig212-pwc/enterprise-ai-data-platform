from common.builders.reader_builder import ReaderBuilder
from common.readers.csv_reader import CSVReader
from common.readers.delta_reader import DeltaReader
from common.readers.kafka_reader import KafkaReader
from common.readers.parquet_reader import ParquetReader


def test_build_kafka():
    config = {"reader": {"type": "kafka"}}
    reader = ReaderBuilder.build(KafkaReader, config)
    assert isinstance(reader, KafkaReader)


def test_build_parquet():
    config = {"reader": {"type": "parquet"}}
    reader = ReaderBuilder.build(ParquetReader, config)
    assert isinstance(reader, ParquetReader)


def test_build_csv():
    config = {"reader": {"type": "csv"}}
    reader = ReaderBuilder.build(CSVReader, config)
    assert isinstance(reader, CSVReader)


def test_build_delta():
    config = {"reader": {"type": "delta"}}
    reader = ReaderBuilder.build(DeltaReader, config)
    assert isinstance(reader, DeltaReader)
