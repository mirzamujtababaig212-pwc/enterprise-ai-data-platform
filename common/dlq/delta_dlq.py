from common.dlq.base_dlq import BaseDLQ


class DeltaDLQ(BaseDLQ):
    def __init__(self, table):
        self.table = table

    def write(self, df):
        if df is None or df.isEmpty():
            return

        (df.write.format("delta").mode("append").saveAsTable(self.table))
