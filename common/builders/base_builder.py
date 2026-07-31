from abc import ABC, abstractmethod


class BaseBuilder(ABC):
    @abstractmethod
    def build(self, cls, config):
        pass
