from common.writers.console_writer import ConsoleWriter
from common.writers.delta_writer import DeltaWriter
from common.writers.fabric_writer import FabricWriter
from common.writers.iceberg_writer import IcebergWriter
from common.writers.parquet_writer import ParquetWriter
from common.writers.postgres_writer import PostgresWriter
from common.writers.s3_writer import S3Writer
from common.writers.snowflake_writer import SnowflakeWriter


WRITER_REGISTRY = {
    "console": ConsoleWriter,
    "delta": DeltaWriter,
    "fabric": FabricWriter,
    "iceberg": IcebergWriter,
    "parquet": ParquetWriter,
    "postgres": PostgresWriter,
    "s3": S3Writer,
    "snowflake": SnowflakeWriter,
}
