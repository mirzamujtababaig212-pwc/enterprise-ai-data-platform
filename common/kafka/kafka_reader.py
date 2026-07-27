from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame

from common.config.settings import Settings
from common.exceptions.kafka import KafkaException
from common.logging.logger import get_logger

logger = get_logger(__name__)

class KafkaReader:
    @staticmethod
    def read_stream(
            spark: SparkSession,
            topic: str=None,
            bootstrap_servers: str=None
    ) -> DataFrame:
        topic=topic or Settings.kafka.TOPIC
        bootstrap_servers = (
                bootstrap_servers
                or Settings.kafka.BOOTSTRAP_SERVERS
        )
        try:
            logger.info(
                    f"Connecting to Kafka topic={topic}"
            )
            df = (
                spark.readStream
                .format("kafka")
                .option(
                    "kafka.bootstrap.servers",
                    bootstrap_servers
                )
                .option(
                    "subscribe",
                    topic
                )
                .option(
                    "startingOffsets",
                    "latest"
                )
                .load()
            )
            logger.info("Kafka stream initialized.")
            return df 
        except Exception as ex:
            logger.error(str(ex))
            raise KafkaException(str(ex))
