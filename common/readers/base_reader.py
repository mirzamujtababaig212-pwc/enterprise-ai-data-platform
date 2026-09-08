from abc import ABC, abstractmethod

from pyspark.sql import DataFrame, SparkSession


class BaseReader(ABC):
    @abstractmethod
    def read(self, spark: SparkSession) -> DataFrame:
        pass

    def read_stream(self, spark: SparkSession) -> DataFrame:
        raise NotImplementedError(f"{self.__class__.__name__} does not support streaming reads.")
