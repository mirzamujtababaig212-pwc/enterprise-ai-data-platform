from common.metrics.base_metrics import BaseMetrics


class ConsoleMetrics(BaseMetrics):
    def record_batch(self, pipeline, batch_id, batch_df, rejected_df=None, duplicate_df=None):
        print("=" * 60)
        print(f"Pipeline : {pipeline}")
        print(f"Batch    : {batch_id}")
        print(f"Rows     : {batch_df.count()}")
        if rejected_df is not None:
            print(f"Rejected : {rejected_df.count()}")
        print("=" * 60)
