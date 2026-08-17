import time

from common.logging.logger import get_logger
from common.readers.base_reader import BaseReader

logger = get_logger(__name__)


class KafkaReader(BaseReader):

    def __init__(self, options):
        self.options = options

    def read(self, spark):

        start = time.time()

        logger.info("Starting Kafka streaming read")

        df = spark.readStream.format("kafka").options(**self.options).load()

        duration = time.time() - start

        logger.info(
            "Kafka read initialized in %.2f sec",
            duration,
        )

        return df
