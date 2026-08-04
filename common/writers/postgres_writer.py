import time

from common.logging.logger import get_logger
from common.writers.base_writer import BaseWriter

logger = get_logger(__name__)


class PostgresWriter(BaseWriter):
    def __init__(self, url, table, properties, mode="append"):
        self.url = url
        self.table = table
        self.properties = properties
        self.mode = mode

    def write_batch(self, df):
        start = time.time()
        (df.write.mode("append").jdbc(url=self.url, table=self.table, properties=self.properties))
        duration = time.time() - start
        rows = df.count()
        logger.info("Rows Written=%s", rows)

        logger.info("Write Duration=%.2f", duration)

    def write_stream(self, df, foreach_batch):

        return df.writeStream.foreachBatch(lambda batch, batch_id: self.write_batch(batch)).start()

    def write(self, df):
        self.write_batch(df)
