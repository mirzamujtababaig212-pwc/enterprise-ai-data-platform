from abc import ABC, abstractmethod


class ReaderStrategy(ABC):
    @abstractmethod
    def read(self, spark):
        pass
