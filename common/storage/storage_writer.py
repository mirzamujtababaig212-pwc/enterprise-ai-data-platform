from __future__ import annotations

from common.logging.logger import get_logger
from common.storage.delta_writer import DeltaWriter
from common.storage.fabric_writer import FabricWriter
from common.storage.postgres_writer import PostgresWriter
from common.storage.snowflake_writer import SnowflakeWriter

logger = get_logger(__name__)


class StorageWriter:
    """
    Backward-compatible storage facade.

    New code should prefer the concrete writer classes or
    WriterBuilder.
    """

    @staticmethod
    def write_stream(
        df,
        target,
        table,
        checkpoint,
        output_mode="append",
        trigger=None,
        foreach_batch=None,
    ):

        target = target.lower()

        if target == "delta":

            if foreach_batch is None:
                raise ValueError("Delta streaming requires " "foreach_batch.")

            writer = DeltaWriter(
                table=table,
                path=table,
                checkpoint=checkpoint,
                mode="append",
            )

            return writer.write_stream(
                df=df,
                foreach_batch=foreach_batch,
                checkpoint=checkpoint,
                output_mode=output_mode,
                trigger=trigger,
            )

        if target == "console":

            writer = (
                df.writeStream.outputMode(output_mode)
                .option(
                    "checkpointLocation",
                    checkpoint,
                )
                .format("console")
            )

            if trigger:
                writer = writer.trigger(**trigger)

            return writer.start()

        if target == "parquet":

            writer = (
                df.writeStream.outputMode(output_mode)
                .option(
                    "checkpointLocation",
                    checkpoint,
                )
                .format("parquet")
                .option(
                    "path",
                    table,
                )
            )

            if trigger:
                writer = writer.trigger(**trigger)

            if foreach_batch:
                writer = writer.foreachBatch(foreach_batch)

            return writer.start()

        if target == "kafka":

            raise NotImplementedError("Kafka streaming writer " "not implemented yet.")

        raise ValueError(f"Unsupported streaming target: " f"{target}")

    @staticmethod
    def write_batch(
        df,
        target: str,
        table: str,
        mode: str = "append",
    ):

        target = target.lower()

        logger.info(
            "Storage target: %s",
            target,
        )

        if target == "postgres":

            return PostgresWriter.write_table(
                df=df,
                table_name=table,
                mode=mode,
            )

        if target == "snowflake":

            return SnowflakeWriter.write_table(
                df=df,
                table_name=table,
                mode=mode,
            )

        if target == "delta":

            writer = DeltaWriter(
                table=table,
                path=table,
                mode=mode,
            )

            return writer.write_batch(df)

        if target == "fabric":

            return FabricWriter.write_table(
                df=df,
                table_name=table,
                mode=mode,
            )

        raise ValueError(f"Unsupported storage target: " f"{target}")
