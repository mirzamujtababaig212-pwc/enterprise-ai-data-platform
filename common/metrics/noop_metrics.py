from common.metrics.base_metrics import BaseMetrics


class NoOpMetrics(BaseMetrics):
    def record_batch(self, pipeline, batch_id, batch_df, rejected_df=None, duplicate_df=None):
        pass
