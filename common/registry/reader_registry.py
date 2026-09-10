from common.readers.csv_reader import CSVReader
from common.readers.delta_reader import DeltaReader
from common.readers.databricks_reader import DatabricksReader
from common.readers.kafka_reader import KafkaReader
from common.readers.parquet_reader import ParquetReader
from common.readers.fabric_reader import FabricReader
from common.readers.postgres_reader import PostgresReader
from common.readers.snowflake_reader import SnowflakeReader

READER_REGISTRY = {
    "kafka": KafkaReader,
    "parquet": ParquetReader,
    "csv": CSVReader,
    "delta": DeltaReader,
    "databricks": DatabricksReader,
    "postgres": PostgresReader,
    "snowflake": SnowflakeReader,
    "fabric": FabricReader,
}
