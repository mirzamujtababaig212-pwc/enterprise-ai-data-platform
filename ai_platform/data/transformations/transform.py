"""Provider-independent data transformation implementations."""

from collections.abc import Callable, Iterable
from typing import Any


Record = dict[str, Any]


class BaseDataTransformer:
    """Base implementation for data transformers."""

    def transform(
        self,
        data: Any,
        **kwargs: Any,
    ) -> Any:
        """Transform input data.

        Subclasses must implement this method.
        """
        raise NotImplementedError


class ColumnRenameTransformer(BaseDataTransformer):
    """Rename columns in a collection of dictionary records."""

    def __init__(
        self,
        mapping: dict[str, str],
    ) -> None:
        self._mapping = dict(mapping)

    def transform(
        self,
        data: Iterable[Record],
        **kwargs: Any,
    ) -> list[Record]:
        """Rename configured columns while preserving all other fields."""

        records = list(data)

        result: list[Record] = []

        for record in records:
            transformed: Record = {}

            for key, value in record.items():
                target_key = self._mapping.get(key, key)
                transformed[target_key] = value

            result.append(transformed)

        return result


class SelectColumnsTransformer(BaseDataTransformer):
    """Select a defined set of columns from dictionary records."""

    def __init__(
        self,
        columns: Iterable[str],
    ) -> None:
        self._columns = tuple(columns)

    def transform(
        self,
        data: Iterable[Record],
        **kwargs: Any,
    ) -> list[Record]:
        """Return records containing only the requested columns."""

        records = list(data)

        return [
            {column: record[column] for column in self._columns if column in record}
            for record in records
        ]


class FilterTransformer(BaseDataTransformer):
    """Filter records using a caller-provided predicate."""

    def __init__(
        self,
        predicate: Callable[[Record], bool],
    ) -> None:
        self._predicate = predicate

    def transform(
        self,
        data: Iterable[Record],
        **kwargs: Any,
    ) -> list[Record]:
        """Return only records satisfying the predicate."""

        records = list(data)

        return [record for record in records if self._predicate(record)]
