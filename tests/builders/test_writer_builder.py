import pytest

from common.builders.writer_builder import WriterBuilder
from common.writers.console_writer import ConsoleWriter
from common.writers.delta_writer import DeltaWriter
from common.writers.fabric_writer import FabricWriter
from common.writers.iceberg_writer import IcebergWriter
from common.writers.parquet_writer import ParquetWriter
from common.writers.postgres_writer import PostgresWriter
from common.writers.s3_writer import S3Writer
from common.writers.snowflake_writer import SnowflakeWriter


def test_build_delta():
    config = {
        "writer": {
            "type": "delta",
        }
    }

    writer = WriterBuilder.build(
        DeltaWriter,
        config,
    )

    assert isinstance(
        writer,
        DeltaWriter,
    )


def test_build_delta_with_explicit_values():
    config = {
        "writer": {
            "type": "delta",
            "table": "bronze.vehicle",
            "path": "/tmp/delta/bronze",
            "checkpoint": "/tmp/checkpoint",
            "mode": "append",
            "output_mode": "append",
        }
    }

    writer = WriterBuilder.build(
        DeltaWriter,
        config,
    )

    assert isinstance(
        writer,
        DeltaWriter,
    )

    assert writer.table == "bronze.vehicle"
    assert str(writer.path) == "/tmp/delta/bronze"
    assert str(writer.checkpoint) == "/tmp/checkpoint"
    assert writer.mode == "append"
    assert writer.output_mode == "append"


def test_build_fabric():
    config = {
        "writer": {
            "type": "fabric",
            "table": "silver.vehicle",
            "checkpoint": "/tmp/fabric-checkpoint",
            "mode": "append",
            "output_mode": "append",
        }
    }

    writer = WriterBuilder.build(
        FabricWriter,
        config,
    )

    assert isinstance(
        writer,
        FabricWriter,
    )

    assert writer.table == "silver.vehicle"
    assert writer.mode == "append"
    assert writer.output_mode == "append"


def test_build_parquet():
    config = {
        "writer": {
            "type": "parquet",
            "path": "/tmp/parquet",
            "mode": "overwrite",
        }
    }

    writer = WriterBuilder.build(
        ParquetWriter,
        config,
    )

    assert isinstance(
        writer,
        ParquetWriter,
    )

    assert writer.path == "/tmp/parquet"
    assert writer.mode == "overwrite"


def test_build_postgres():
    config = {
        "writer": {
            "type": "postgres",
            "url": "jdbc:postgresql://localhost/test",
            "table": "vehicle",
            "properties": {},
            "mode": "append",
        }
    }

    writer = WriterBuilder.build(
        PostgresWriter,
        config,
    )

    assert isinstance(
        writer,
        PostgresWriter,
    )

    assert writer.url == "jdbc:postgresql://localhost/test"
    assert writer.table == "vehicle"
    assert writer.properties == {}
    assert writer.mode == "append"


def test_build_snowflake():
    config = {
        "writer": {
            "type": "snowflake",
            "options": {
                "sfURL": "example.snowflakecomputing.com",
                "sfDatabase": "TEST",
            },
            "table": "VEHICLE",
            "mode": "append",
        }
    }

    writer = WriterBuilder.build(
        SnowflakeWriter,
        config,
    )

    assert isinstance(
        writer,
        SnowflakeWriter,
    )

    assert writer.options["sfURL"] == "example.snowflakecomputing.com"
    assert writer.table == "VEHICLE"
    assert writer.mode == "append"


def test_build_s3():
    config = {
        "writer": {
            "type": "s3",
            "path": "/tmp/output",
        }
    }

    writer = WriterBuilder.build(
        S3Writer,
        config,
    )

    assert isinstance(
        writer,
        S3Writer,
    )

    assert writer.path == "/tmp/output"


def test_build_iceberg():
    config = {
        "writer": {
            "type": "iceberg",
            "table": "catalog.db.vehicle",
        }
    }

    writer = WriterBuilder.build(
        IcebergWriter,
        config,
    )

    assert isinstance(
        writer,
        IcebergWriter,
    )

    assert writer.table == "catalog.db.vehicle"


def test_build_console():
    config = {
        "writer": {
            "type": "console",
        }
    }

    writer = WriterBuilder.build(
        ConsoleWriter,
        config,
    )

    assert isinstance(
        writer,
        ConsoleWriter,
    )


def test_empty_config():
    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        WriterBuilder.build(
            ConsoleWriter,
            {},
        )


def test_missing_writer_type():
    config = {"writer": {}}

    with pytest.raises(
        ValueError,
        match="Writer type is required",
    ):
        WriterBuilder.build(
            ConsoleWriter,
            config,
        )


def test_missing_writer_class():
    config = {
        "writer": {
            "type": "delta",
        }
    }

    with pytest.raises(
        ValueError,
        match="Writer class is required",
    ):
        WriterBuilder.build(
            None,
            config,
        )


def test_invalid_writer_type():
    config = {
        "writer": {
            "type": "dummy",
        }
    }

    with pytest.raises(
        ValueError,
        match="Unsupported writer type",
    ):
        WriterBuilder.build(
            ConsoleWriter,
            config,
        )
