from common.writers.console_writer import ConsoleWriter
from common.writers.delta_writer import DeltaWriter
from common.writers.fabric_writer import FabricWriter
from common.writers.kafka_writer import KafkaWriter
from common.writers.parquet_writer import ParquetWriter
from common.writers.postgres_writer import PostgresWriter
from common.writers.snowflake_writer import SnowflakeWriter


class StorageWriter:
    @staticmethod
    def write(df, target, table, mode="append"):
        if target == "postgres":
            PostgresWriter(url=..., table=table, properties=..., mode=mode).write_batch(
                df
            )
        elif target == "snowflake":
            SnowflakeWriter(options=..., table=table, mode=mode).write_batch(df)
        elif target == "delta":
            DeltaWriter(table=table, checkpoint=None, mode=mode).write_batch(df)
        elif target == "fabric":
            FabricWriter(table=table, mode=mode).write_batch(df)
        elif target == "Parquet":
            ParquetWriter(path=table, mode=mode).write_batch(df)
        elif target == "Kafka":
            KafkaWriter.write_table(df, table, mode)
        elif target == "console":
            ConsoleWriter().write_batch(df)
        else:
            raise Exception(f"Unknown target {target}")
