from common.logging.logger import get_logger

logger = get_logger(__name__)


class MetricsCollector:
    @staticmethod
    def log_batch_metrics(valid_df, invalid_df):
        valid = valid_df.count()
        invalid = invalid_df.count()
        logger.info(f"Valid Records: {valid}")
        logger.info(f"Rejected Records: {invalid}")

    @staticmethod
    def record(metric_name, value):

        logger.info(f"Metric | {metric_name} = {value}")
