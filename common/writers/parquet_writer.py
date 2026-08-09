import time

from common.logging.logger import get_logger
from common.writers.base_writer import BaseWriter

logger = get_logger(__name__)


class ParquetWriter(BaseWriter):
    def __init__(self, path, mode="append"):
        self.path = path
        self.mode = mode

    def write_batch(self, df):
        start = time.time()
        (df.write.mode(self.mode).parquet(self.path))
        duration = time.time() - start
        logger.info("Rows Written=%s", df.count())
        logger.info("Write Duration=%.2f sec", duration)

    def write_stream(self, df, foreach_batch):
        return df.writeStream.foreachBatch(lambda batch, _: self.write_batch(batch)).start()

    def write(self, df):
        self.write_batch(df)
