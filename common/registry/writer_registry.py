from common.writers.console_writer import ConsoleWriter
from common.writers.delta_writer import DeltaWriter
from common.writers.iceberg_writer import IcebergWriter
from common.writers.postgres_writer import PostgresWriter
from common.writers.s3_writer import S3Writer

WRITER_REGISTRY = {
    "delta": DeltaWriter,
    "postgres": PostgresWriter,
    "console": ConsoleWriter,
    "iceberg": IcebergWriter,
    "s3": S3Writer,
}
