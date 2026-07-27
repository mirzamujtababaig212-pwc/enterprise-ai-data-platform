import time

from common.logging.logger import get_logger
from common.metrics.base_metrics import BaseMetrics

logger = get_logger(__name__)

class MetricsCollector(BaseMetrics):
    def record_batch(
        self,
            pipeline,
            batch_id,
            batch_df,
            rejected_df=None,
            duplicate_df=None
    ):
        start = time.time()
        processed = batch_df.cache().count()
        rejected = (
            rejected_df.cache().count()
            if rejected_df is not None
            else 0
        )
        duplicates = (
            duplicate_df.cache().count()
            if duplicate_df is not None
            else 0
        )
        latency = round(time.time() - start, 2)
        logger.info(
            "%s Batch %s",
            pipeline,
            batch_id
        )

        logger.info(
            "Rows=%s",
            processed
        )
        if rejected_df is not None:
            logger.info(
                "Rejected=%s",
                rejected
            )
        logger.info(
            """
        Pipeline : %s
        Batch Id : %s
        Processed : %s
        Rejected : %s
        Duplicates : %s
        Latency : %s sec
        """,
            pipeline,
            batch_id,
            processed,
            rejected,
            duplicates,
            latency
        )
        return {
            "processed": processed,
            "rejected": rejected,
            "duplicates": duplicates,
            "latency": latency
        }
