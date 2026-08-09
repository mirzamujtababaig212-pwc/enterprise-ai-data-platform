from abc import ABC, abstractmethod


class BaseValidator(ABC):
    @abstractmethod
    def validate(self, df):
        """
        Returns:
            valid_df,
            invalid_df
        """
        ...
