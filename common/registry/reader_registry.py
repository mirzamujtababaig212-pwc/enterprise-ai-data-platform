from common.readers.csv_reader import CSVReader
from common.readers.delta_reader import DeltaReader
from common.readers.kafka_reader import KafkaReader
from common.readers.parquet_reader import ParquetReader

READER_REGISTRY = {
    "kafka": KafkaReader,
    "parquet": ParquetReader,
    "csv": CSVReader,
    "delta": DeltaReader,
}
