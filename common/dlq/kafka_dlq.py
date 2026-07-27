from common.dlq.base_dlq import BaseDLQ


class KafkaDLQ(BaseDLQ):
    def write(self, df):
        raise NotImplementedError("KafkaDLQ not implemented yet")
