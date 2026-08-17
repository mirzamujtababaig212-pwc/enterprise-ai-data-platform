"""Reusable provider-independent data quality rules."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

Record = dict[str, Any]


@dataclass(frozen=True)
class QualityResult:
    """Standard result produced by a data quality rule."""

    rule_name: str
    passed: bool
    checked_records: int
    failed_records: int
    message: str

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable quality result."""

        return {
            "rule_name": self.rule_name,
            "passed": self.passed,
            "checked_records": self.checked_records,
            "failed_records": self.failed_records,
            "message": self.message,
        }


class BaseQualityRule:
    """Base implementation for data quality rules."""

    rule_name: str = "base"

    def evaluate(
        self,
        data: Any,
    ) -> dict[str, Any]:
        """Evaluate the quality rule."""

        raise NotImplementedError


class NotNullRule(BaseQualityRule):
    """Verify that a field is present and non-null."""

    def __init__(
        self,
        column: str,
    ) -> None:
        if not column:
            raise ValueError("NotNullRule requires a column name.")

        self.column = column
        self.rule_name = f"not_null:{column}"

    def evaluate(
        self,
        data: Iterable[Record],
    ) -> dict[str, Any]:
        """Check that the configured column contains non-null values."""

        records = list(data)

        failed_records = sum(
            1 for record in records if self.column not in record or record[self.column] is None
        )

        result = QualityResult(
            rule_name=self.rule_name,
            passed=failed_records == 0,
            checked_records=len(records),
            failed_records=failed_records,
            message=(
                f"Column '{self.column}' passed not-null validation."
                if failed_records == 0
                else (
                    f"Column '{self.column}' contains " f"{failed_records} null or missing values."
                )
            ),
        )

        return result.to_dict()


class UniqueRule(BaseQualityRule):
    """Verify that values in a field are unique."""

    def __init__(
        self,
        column: str,
    ) -> None:
        if not column:
            raise ValueError("UniqueRule requires a column name.")

        self.column = column
        self.rule_name = f"unique:{column}"

    def evaluate(
        self,
        data: Iterable[Record],
    ) -> dict[str, Any]:
        """Check uniqueness of the configured column."""

        records = list(data)

        values: list[Any] = []

        for record in records:
            if self.column in record:
                values.append(record[self.column])

        unique_values = len(set(values))
        failed_records = len(values) - unique_values

        result = QualityResult(
            rule_name=self.rule_name,
            passed=failed_records == 0,
            checked_records=len(records),
            failed_records=failed_records,
            message=(
                f"Column '{self.column}' contains unique values."
                if failed_records == 0
                else (
                    f"Column '{self.column}' contains "
                    f"{failed_records} duplicate value occurrence(s)."
                )
            ),
        )

        return result.to_dict()


class AllowedValuesRule(BaseQualityRule):
    """Verify that a field contains only allowed values."""

    def __init__(
        self,
        column: str,
        allowed_values: Iterable[Any],
    ) -> None:
        if not column:
            raise ValueError("AllowedValuesRule requires a column name.")

        self.column = column
        self.allowed_values = frozenset(allowed_values)
        self.rule_name = f"allowed_values:{column}"

    def evaluate(
        self,
        data: Iterable[Record],
    ) -> dict[str, Any]:
        """Check that every configured value is allowed."""

        records = list(data)

        failed_records = sum(
            1
            for record in records
            if (self.column not in record or record[self.column] not in self.allowed_values)
        )

        result = QualityResult(
            rule_name=self.rule_name,
            passed=failed_records == 0,
            checked_records=len(records),
            failed_records=failed_records,
            message=(
                f"Column '{self.column}' contains only allowed values."
                if failed_records == 0
                else (f"Column '{self.column}' contains " f"{failed_records} invalid value(s).")
            ),
        )

        return result.to_dict()


class NumericRangeRule(BaseQualityRule):
    """Verify that numeric values fall within an inclusive range."""

    def __init__(
        self,
        column: str,
        minimum: float | None = None,
        maximum: float | None = None,
    ) -> None:
        if not column:
            raise ValueError("NumericRangeRule requires a column name.")

        if minimum is None and maximum is None:
            raise ValueError("NumericRangeRule requires minimum or maximum.")

        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError("NumericRangeRule minimum cannot exceed maximum.")

        self.column = column
        self.minimum = minimum
        self.maximum = maximum
        self.rule_name = f"numeric_range:{column}"

    def evaluate(
        self,
        data: Iterable[Record],
    ) -> dict[str, Any]:
        """Check that numeric values are inside the configured range."""

        records = list(data)
        failed_records = 0

        for record in records:
            if self.column not in record:
                failed_records += 1
                continue

            value = record[self.column]

            if not isinstance(value, (int, float)):
                failed_records += 1
                continue

            if self.minimum is not None and value < self.minimum:
                failed_records += 1
                continue

            if self.maximum is not None and value > self.maximum:
                failed_records += 1

        result = QualityResult(
            rule_name=self.rule_name,
            passed=failed_records == 0,
            checked_records=len(records),
            failed_records=failed_records,
            message=(
                f"Column '{self.column}' passed numeric range validation."
                if failed_records == 0
                else (
                    f"Column '{self.column}' contains "
                    f"{failed_records} value(s) outside the "
                    "configured numeric range."
                )
            ),
        )

        return result.to_dict()
