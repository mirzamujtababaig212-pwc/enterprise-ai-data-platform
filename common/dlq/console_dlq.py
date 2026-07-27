from common.dlq.base_dlq import BaseDLQ


class ConsoleDLQ(BaseDLQ):
    def write(self, df):
        df.show(
            truncate=False
        )
