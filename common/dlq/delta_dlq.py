from common.dlq.base_dlq import BaseDLQ


class DeltaDLQ(BaseDLQ):
    def __init__(self, table):
        self.table = table

    def write(self, df):
        (
            df.write
              .format("delta")
              .mode("append")
              .saveAsTable(self.table)
        )
