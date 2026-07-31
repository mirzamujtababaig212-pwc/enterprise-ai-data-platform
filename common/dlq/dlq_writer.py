import time

from common.logging.logger import get_logger
from common.storage.storage_writer import StorageWriter

logger = get_logger(__name__)


class DLQWriter:

    @staticmethod
    def write(df, target="delta", table="vehicle_dlq", mode="append"):
        start = time.time()
        StorageWriter.write_batch(df=df, target=target, table=table, mode=mode)
        duration = time.time() - start
        logger.info("DLQ Records=%s", df.count())
        logger.info("DLQ Write Time=%.2f sec", duration)
