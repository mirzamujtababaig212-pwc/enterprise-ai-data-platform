from common.config.database import PostgresConfig
from common.config.databricks import DatabricksConfig
from common.config.environment import Environment
from common.config.kafka import KafkaConfig
from common.config.spark import SparkConfig
from common.config.storage import StorageConfig
from common.config.snowflake import SnowflakeConfig
from common.config.fabric import FabricConfig
from common.config.qdrant import QdrantConfig
from common.config.vector_store import VectorStoreConfig
from common.config.memory import MemoryStoreConfig


class Settings:
    postgres = PostgresConfig
    kafka = KafkaConfig
    spark = SparkConfig
    env = Environment
    storage = StorageConfig()
    snowflake = SnowflakeConfig
    databricks = DatabricksConfig
    fabric = FabricConfig
    qdrant = QdrantConfig
    vector_store = VectorStoreConfig
    memory_store = MemoryStoreConfig
