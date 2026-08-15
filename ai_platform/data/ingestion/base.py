from abc import ABC, abstractmethod
from typing import Any
from ai_platform.data.contracts import DataSource


class BaseDataReader(ABC):
    """
    Base implementation for data readers.

    Concrete readers implement the read operation while sharing the same
    DataReader architectural contract.
    """

    @abstractmethod
    def read(
        self,
        source: DataSource,
        **kwargs: Any,
    ) -> Any:
        """Read data from the supplied source."""
        raise NotImplementedError
