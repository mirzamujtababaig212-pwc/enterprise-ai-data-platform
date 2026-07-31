from common.kafka.kafka_reader import KafkaReader
from common.readers.delta_reader import DeltaReader
from common.readers.fabric_reader import FabricReader
from common.readers.parquet_reader import ParquetReader
from common.readers.postgres_reader import PostgresReader
from common.readers.snowflake_reader import SnowflakeReader


class SparkReader:
    @staticmethod
    def read_kafka(spark, topic=None, bootstrap_servers=None):
        return KafkaReader.read_stream(spark, topic, bootstrap_servers)

    @staticmethod
    def read_parquet(spark, path, schema=None):
        reader = ParquetReader(path, schema)
        return reader.read(spark)

    @staticmethod
    def read_delta(spark, path):
        reader = DeltaReader(path)
        return reader.read(spark)

    @staticmethod
    def read_postgres(spark, table):
        return PostgresReader.read_table(spark, table)

    @staticmethod
    def read_snowflake(spark, table):
        return SnowflakeReader.read_table(spark, table)

    @staticmethod
    def read_fabric(spark, table):
        return FabricReader.read_table(spark, table)
