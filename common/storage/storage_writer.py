from common.logging.logger import get_logger
from common.storage.delta_writer import DeltaWriter
from common.storage.fabric_writer import FabricWriter
from common.storage.postgres_writer import PostgresWriter
from common.storage.snowflake_writer import SnowflakeWriter

logger = get_logger(__name__)

class StorageWriter:
    @staticmethod
    def write_stream(
        df,
        target,
        table,
        checkpoint,
        output_mode="append",
        trigger=None,
        foreach_batch=None
    ):
       writer = (
       df.writeStream
         .outputMode(output_mode)
         .option(
            "checkpointLocation",
            checkpoint
         )
       )
       if trigger:
            writer = writer.trigger(**trigger)
       if foreach_batch:
            writer = writer.foreachBatch(
                foreach_batch
            )
       if target == "delta":
            return (
                writer
                .toTable(table)
            )
       elif target == "console":
            return (
                writer
                .format("console")
                .start()
            )
       elif target == "parquet":
            return (
                writer
                .format("parquet")
                .option("path", table)
                .start()
            )
       elif target == "kafka":
            raise NotImplementedError("Kafka streaming writer not implemented yet")

    @staticmethod
    def write_batch(
        df,
        target: str,
        table: str,
        mode: str = "append"
    ):
        target = target.lower()
        logger.info(f"Storage target: {target}")
        if target == "postgres":
            return PostgresWriter.write_table(
                df=df,
                table_name=table,
                mode=mode
            )
        elif target == "snowflake":
            return SnowflakeWriter.write_table(
                df=df,
                table_name=table,
                mode=mode
            )
        elif target == "delta":
            return DeltaWriter.write_table(
                df=df,
                table_name=table,
                mode=mode
            )
        elif target == "fabric":
            return FabricWriter.write_table(
                df=df,
                table_name=table,
                mode=mode
            )
        else:
            raise ValueError(
                f"Unsupported storage target: {target}"
            )
