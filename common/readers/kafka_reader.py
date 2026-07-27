from common.logging.logger import get_logger
from common.readers.base_reader import BaseReader

logger = get_logger(__name__)

class KafkaReader(BaseReader):

    def __init__(self, options):
        self.options = options

    def read(self, spark):
        return (
            spark.readStream
                 .format("kafka")
                 .options(**self.options)
                 .load()
        )

