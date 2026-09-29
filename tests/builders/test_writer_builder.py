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

    writer = WriterBuilder.build(config)

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

    writer = WriterBuilder.build(config)

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

    writer = WriterBuilder.build(config)

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

    writer = WriterBuilder.build(config)

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

    writer = WriterBuilder.build(config)

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

    writer = WriterBuilder.build(config)

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

    writer = WriterBuilder.build(config)

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

    writer = WriterBuilder.build(config)

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

    writer = WriterBuilder.build(config)

    assert isinstance(
        writer,
        ConsoleWriter,
    )


def test_empty_config():
    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        WriterBuilder.build({})


def test_missing_writer_type():
    config = {"writer": {}}

    with pytest.raises(
        ValueError,
        match="Writer type is required",
    ):
        WriterBuilder.build(config)


def test_build_resolves_writer_class_from_registry():
    config = {
        "writer": {
            "type": "console",
        }
    }

    writer = WriterBuilder.build(config)

    assert isinstance(writer, ConsoleWriter)


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
        WriterBuilder.build(config)


def test_snowflake_writer_requires_table():

    import pytest

    from common.builders.writer_builder import (
        WriterBuilder,
    )

    config = {
        "writer": {
            "type": "snowflake",
        }
    }

    with pytest.raises(
        ValueError,
        match="Snowflake writer requires a table",
    ):
        WriterBuilder.build(config)


def test_build_delta_merge_with_keys():
    config = {
        "writer": {
            "type": "delta",
            "table": "silver.vehicle",
            "path": "/tmp/delta/silver",
            "mode": "merge",
            "merge_keys": [
                "vehicle_id",
                "event_time",
            ],
        }
    }

    writer = WriterBuilder.build(config)

    assert isinstance(
        writer,
        DeltaWriter,
    )

    assert writer.table == "silver.vehicle"
    assert str(writer.path) == "/tmp/delta/silver"
    assert writer.mode == "merge"
    assert writer.merge_keys == [
        "vehicle_id",
        "event_time",
    ]


def test_build_delta_accepts_optional_glue_configuration():
    config = {
        "writer": {
            "type": "delta",
            "table": "bronze.vehicle_events",
            "path": "/tmp/delta/bronze",
            "glue_database_name": "enterprise_ai_platform",
            "glue_table_name": "vehicle_events",
        }
    }

    writer = WriterBuilder.build(config)

    assert isinstance(writer, DeltaWriter)
    assert writer._glue_database_name == "enterprise_ai_platform"
    assert writer._glue_table_name == "vehicle_events"
    assert writer._glue_synchronizer is None


def test_delta_rejects_glue_without_database_name():
    with pytest.raises(
        ValueError,
        match="Glue database name is required",
    ):
        DeltaWriter(
            table="bronze.vehicle_events",
            path="/tmp/delta/bronze",
            glue_synchronizer=object(),
        )


def test_build_delta_accepts_injected_glue_synchronizer():
    synchronizer = object()

    config = {
        "writer": {
            "type": "delta",
            "table": "bronze.vehicle_events",
            "path": "s3a://enterprise-data-ai-platform/bronze/vehicle_events",
            "glue_database_name": "enterprise_ai_platform",
            "glue_table_name": "vehicle_events",
        }
    }

    writer = WriterBuilder.build(
        config,
        glue_synchronizer=synchronizer,
    )

    assert writer._glue_synchronizer is synchronizer
