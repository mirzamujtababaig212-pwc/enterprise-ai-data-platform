import time

from common.logging.logger import get_logger
from common.metrics.base_metrics import BaseMetrics

logger = get_logger(__name__)


def calculate_status(processed, rejected):
    if processed == 0:
        return "FAILED"
    if rejected > 0:
        return "PARTIAL_SUCCESS"
    return "SUCCESS"


class MetricsCollector(BaseMetrics):
    def record_batch(
        self,
        pipeline,
        batch_id,
        batch_df,
        rejected_df=None,
        duplicate_df=None,
        retry_count=0,
        transform_duration=0,
        validation_duration=0,
        write_duration=0,
        dlq_duration=0,
        pipeline_duration=0,
    ):
        start = time.time()
        processed = batch_df.cache().count()
        rejected = rejected_df.cache().count() if rejected_df is not None else 0
        duplicates = duplicate_df.cache().count() if duplicate_df is not None else 0
        status = calculate_status(processed, rejected)
        pipeline_duration = round(time.time() - start, 2)
        throughput = processed / pipeline_duration if pipeline_duration > 0 else processed
        success_rate = (
            ((processed - rejected) / processed * 100) if (processed + rejected) > 0 else 0.0
        )
        logger.info("%s Batch %s", pipeline, batch_id)
        logger.info("Transform Duration : %.2f", transform_duration)
        logger.info("Validation Duration : %.2f", validation_duration)
        logger.info("Write Duration : %.2f", write_duration)
        logger.info("DLQ Duration : %.2f", dlq_duration)
        logger.info("Pipeline Duration : %.2f", pipeline_duration)
        logger.info("Rows=%s", processed)
        logger.info("Pipeline Status=%s", status)
        if rejected_df is not None:
            logger.info("Rejected=%s", rejected)
        logger.info(
            """
        Pipeline : %s
        Batch Id : %s
        Processed : %s
        Rejected : %s
        Duplicates : %s
        Pipeline Duration : %.2f sec
        Throughput : %s
        Status : %s
        Success Rate: %s
        """,
            pipeline,
            batch_id,
            processed,
            rejected,
            duplicates,
            pipeline_duration,
            throughput,
            status,
            success_rate,
        )
        return {
            "processed": processed,
            "rejected": rejected,
            "duplicates": duplicates,
            "pipeline_duration": pipeline_duration,
            "throughput": throughput,
            "status": status,
            "retry_count": retry_count,
            "success_rate": success_rate,
            "transform_duration": transform_duration,
            "validation_duration": validation_duration,
            "write_duration": write_duration,
            "dlq_duration": dlq_duration,
        }
