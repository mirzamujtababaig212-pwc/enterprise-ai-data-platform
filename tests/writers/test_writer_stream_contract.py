import inspect

from common.writers.base_writer import BaseWriter
from common.writers.console_writer import ConsoleWriter
from common.writers.fabric_writer import FabricWriter
from common.writers.iceberg_writer import IcebergWriter
from common.writers.parquet_writer import ParquetWriter
from common.writers.postgres_writer import PostgresWriter
from common.writers.s3_writer import S3Writer
from common.writers.snowflake_writer import SnowflakeWriter
from common.writers.delta_writer import DeltaWriter

EXPECTED_PARAMETERS = [
    "self",
    "df",
    "foreach_batch",
    "checkpoint",
    "output_mode",
    "query_name",
    "trigger",
]


def test_base_writer_stream_contract():
    signature = inspect.signature(BaseWriter.write_stream)

    assert list(signature.parameters) == EXPECTED_PARAMETERS


def test_all_writers_expose_canonical_stream_contract():
    writers = [
        ConsoleWriter,
        DeltaWriter,
        FabricWriter,
        IcebergWriter,
        ParquetWriter,
        PostgresWriter,
        S3Writer,
        SnowflakeWriter,
    ]

    for writer_cls in writers:
        signature = inspect.signature(writer_cls.write_stream)

        assert list(signature.parameters) == EXPECTED_PARAMETERS
