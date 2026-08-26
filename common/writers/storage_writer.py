from __future__ import annotations

from common.writers.console_writer import ConsoleWriter
from common.writers.delta_writer import DeltaWriter
from common.writers.fabric_writer import FabricWriter
from common.writers.parquet_writer import ParquetWriter


class StorageWriter:
    """
    Canonical storage writer facade.

    Delegates DataFrame writes to concrete implementations under
    ``common.writers``.

    Configuration-independent targets:
        - delta
        - fabric
        - parquet
        - console

    Database-specific writers such as Postgres and Snowflake require
    connection configuration and should be constructed through
    dependency injection or WriterBuilder.
    """

    SUPPORTED_TARGETS = {
        "delta",
        "fabric",
        "parquet",
        "console",
    }

    @staticmethod
    def write(
        df,
        target,
        table,
        mode="append",
    ):
        """
        Write a DataFrame to the requested storage target.

        Args:
            df: Spark DataFrame.
            target: Storage target name.
            table: Target table name or path.
            mode: Spark write mode.

        Returns:
            Result returned by the concrete writer.

        Raises:
            ValueError: If target is missing, unsupported, or requires
                configuration unavailable to this facade.
        """

        if not target:
            raise ValueError("Storage writer target is required.")

        target = target.strip().lower()

        if target == "delta":
            return DeltaWriter(
                table=table,
                checkpoint=None,
                mode=mode,
            ).write_batch(df)

        if target == "fabric":
            return FabricWriter(
                table=table,
                mode=mode,
            ).write_batch(df)

        if target == "parquet":
            return ParquetWriter(
                path=table,
                mode=mode,
            ).write_batch(df)

        if target == "console":
            return ConsoleWriter().write_batch(df)

        if target in {
            "postgres",
            "snowflake",
        }:
            raise ValueError(
                f"{target} requires connection configuration and "
                "must be constructed through dependency injection "
                "or WriterBuilder."
            )

        raise ValueError(f"Unknown storage writer target: {target}")
