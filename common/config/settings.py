from common.config.database import PostgresConfig
from common.config.environment import Environment
from common.config.kafka import KafkaConfig
from common.config.spark import SparkConfig
from common.config.storage import StorageConfig
from common.config.snowflake import SnowflakeConfig
from common.config.fabric import FabricConfig
from common.config.qdrant import QdrantConfig


class Settings:
    postgres = PostgresConfig
    kafka = KafkaConfig
    spark = SparkConfig
    env = Environment
    storage = StorageConfig
    snowflake = SnowflakeConfig
    fabric = FabricConfig
    qdrant = QdrantConfig
