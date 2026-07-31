from common.logging.logger import get_logger
from common.metrics.base_metrics import BaseMetrics

logger = get_logger(__name__)


class AzureMonitorMetrics(BaseMetrics):
    def record_batch(
        self,
        pipeline,
        batch_id,
        batch_df,
        rejected_df=None,
        duplicate_df=None,
    ):
        rows = batch_df.count()

        logger.info(
            "Pipeline=%s Batch=%s Rows=%s",
            pipeline,
            batch_id,
            rows,
        )
