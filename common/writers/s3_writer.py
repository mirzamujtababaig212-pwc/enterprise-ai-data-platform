import time

from common.logging.logger import get_logger
from common.writers.base_writer import BaseWriter

logger = get_logger(__name__)


class S3Writer(BaseWriter):
    def __init__(self, path):
        self.path = path

    def write_batch(self, df):
        start = time.time()
        (df.write.mode("append").parquet(self.path))
        duration = time.time() - start
        rows = df.count()
        logger.info("Rows Written=%s", df.count())

        logger.info("Write Duration=%.2f sec", duration)

    def write_stream(self, df, foreach_batch):
        return df.writeStream.foreachBatch(foreach_batch).start()

    def write(self, df):
        self.write_batch(df)
