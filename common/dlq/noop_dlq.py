from common.dlq.base_dlq import BaseDLQ


class NoOpDLQ(BaseDLQ):
    def write(self, df):
        pass
