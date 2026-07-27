from abc import ABC, abstractmethod


class BaseDLQ(ABC):
    @abstractmethod
    def write(self, df):
        pass
