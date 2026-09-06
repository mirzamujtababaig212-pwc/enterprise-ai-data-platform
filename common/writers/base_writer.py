from abc import ABC, abstractmethod

from pyspark.sql import DataFrame


class BaseWriter(ABC):
    @abstractmethod
    def write(self, df: DataFrame):
        pass

    @abstractmethod
    def write_batch(self, df: DataFrame):
        pass

    @abstractmethod
    def write_stream(
        self,
        df: DataFrame,
        foreach_batch,
        checkpoint=None,
        output_mode=None,
        query_name=None,
        trigger=None,
    ):
        pass
