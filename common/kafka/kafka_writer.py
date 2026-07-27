from common.exceptions.kafka import KafkaException
from common.logging.logger import get_logger

logger = get_logger(__name__)

class KafkaWriter:
    @staticmethod
    def write_stream(df, topic, checkpoint):
        try:
            logger.info(
                f"Writing stream to Kafka topic={topic}"
            )
            return (
                df.writeStream
                .format("kafka")
                .option("topic", topic)
                .option(
                    "checkpointLocation",
                    checkpoint
                )
            )
        except Exception as ex:
            logger.error(str(ex))
            raise KafkaException(str(ex))
