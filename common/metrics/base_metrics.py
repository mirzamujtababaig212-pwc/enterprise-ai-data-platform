from abc import ABC, abstractmethod


class BaseMetrics(ABC):
    @abstractmethod
    def record_batch(
        self,
        pipeline,
        batch_id,
        batch_df,
        rejected_df=None,
        duplicate_df=None
    ):
        pass
