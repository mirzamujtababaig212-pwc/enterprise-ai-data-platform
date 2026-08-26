from common.registry.writer_registry import WRITER_REGISTRY

from common.writers.console_writer import ConsoleWriter
from common.writers.delta_writer import DeltaWriter
from common.writers.fabric_writer import FabricWriter
from common.writers.iceberg_writer import IcebergWriter
from common.writers.parquet_writer import ParquetWriter
from common.writers.postgres_writer import PostgresWriter
from common.writers.s3_writer import S3Writer
from common.writers.snowflake_writer import SnowflakeWriter


EXPECTED_WRITERS = {
    "console": ConsoleWriter,
    "delta": DeltaWriter,
    "fabric": FabricWriter,
    "iceberg": IcebergWriter,
    "parquet": ParquetWriter,
    "postgres": PostgresWriter,
    "s3": S3Writer,
    "snowflake": SnowflakeWriter,
}


def test_registry_contains_all_writer_types():
    assert set(WRITER_REGISTRY) == set(EXPECTED_WRITERS)


def test_registry_maps_types_to_correct_classes():
    for writer_type, writer_cls in EXPECTED_WRITERS.items():
        assert WRITER_REGISTRY[writer_type] is writer_cls


def test_registry_values_are_classes():
    for writer_cls in WRITER_REGISTRY.values():
        assert callable(writer_cls)
