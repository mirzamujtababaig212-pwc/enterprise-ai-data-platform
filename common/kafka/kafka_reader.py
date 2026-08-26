from pyspark.sql import SparkSession

from common.config.settings import Settings
from common.exceptions.kafka import KafkaException
from common.logging.logger import get_logger

logger = get_logger(__name__)


class KafkaReader:
    def __init__(self, options=None):
        self.options = options or {}

    def read(self, spark):
        return self.read_stream(
            spark=spark,
            topic=self.options.get("subscribe"),
            bootstrap_servers=self.options.get("kafka.bootstrap.servers"),
        )

    @staticmethod
    def read_stream(
        spark: SparkSession,
        topic: str | None = None,
        bootstrap_servers: str | None = None,
    ):
        topic = topic or Settings.kafka.TOPIC
        bootstrap_servers = bootstrap_servers or Settings.kafka.BOOTSTRAP_SERVERS
        try:
            logger.info(f"Connecting to Kafka topic={topic}")
            df = (
                spark.readStream.format("kafka")
                .option("kafka.bootstrap.servers", bootstrap_servers)
                .option("subscribe", topic)
                .option(
                    "startingOffsets",
                    Settings.kafka.STARTING_OFFSETS,
                )
                .option(
                    "failOnDataLoss",
                    Settings.kafka.FAIL_ON_DATA_LOSS,
                )
                .load()
            )
            logger.info("Kafka stream initialized.")
            return df
        except Exception as ex:
            logger.error(str(ex))
            raise KafkaException(str(ex)) from ex
