from common.writers.delta_writer import DeltaWriter
from common.writers.fabric_writer import FabricWriter
from common.writers.postgres_writer import PostgresWriter
from common.writers.snowflake_writer import SnowflakeWriter


class StorageWriter:
    @staticmethod
    def write(
        df,
        target,
        table,
        mode="append"
    ):
        if target == "postgres":
            PostgresWriter.write_table(
                df,
                table,
                mode
            )
        elif target == "snowflake":
            SnowflakeWriter.write_table(
                df,
                table,
                mode
            )
        elif target == "delta":
            DeltaWriter.write_table(
                df,
                table,
                mode
            )
        elif target == "fabric":
            FabricWriter.write_table(
                df,
                table,
                mode
            )
        elif target == "Parquet":
            ParquetWriter.write_table(
                df,
                table,
                mode
            )
        elif target == "Kafka":
            KafkaWrite.write_table(
                df,
                table,
                mode
            )
        elif target == "console":
            ConsoleWrite.write_table(
                df,
                table,
                mode
            )
        else:
            raise Exception(
                f"Unknown target {target}"
            )
