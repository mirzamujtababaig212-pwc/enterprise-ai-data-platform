from common.readers.csv_reader import CSVReader
from common.readers.delta_reader import DeltaReader
from common.readers.kafka_reader import KafkaReader
from common.readers.parquet_reader import ParquetReader

READER_REGISTRY = {
    "kafka": KafkaReader,
    "parquet": ParquetReader,
    "delta": DeltaReader,
    "csv": CSVReader,
}


class ReaderFactory:

    @staticmethod
    def create(config):

        reader_cfg = config["reader"]

        reader_type = reader_cfg["type"]

        if reader_type not in READER_REGISTRY:
            raise ValueError(f"Unknown reader type: {reader_type}")

        reader_cls = READER_REGISTRY[reader_type]

        kwargs = {key: value for key, value in reader_cfg.items() if key not in {"type", "table"}}

        return reader_cls(**kwargs)
