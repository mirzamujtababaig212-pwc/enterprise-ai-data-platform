from __future__ import annotations

from dataclasses import dataclass

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from common.logging.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """
    Result of a single data-quality check.
    """

    name: str
    passed: bool
    actual: int | float | str
    expected: str
    message: str


class DataQualityValidator:
    """
    Reusable Spark DataFrame data-quality validator.

    The validator performs common checks without modifying
    the underlying DataFrame.
    """

    def __init__(
        self,
        dataframe: DataFrame,
        dataset_name: str,
    ) -> None:

        self.df = dataframe
        self.dataset_name = dataset_name
        self.results: list[ValidationResult] = []

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _record(
        self,
        name: str,
        passed: bool,
        actual: int | float | str,
        expected: str,
        message: str,
    ) -> None:

        result = ValidationResult(
            name=name,
            passed=passed,
            actual=actual,
            expected=expected,
            message=message,
        )

        self.results.append(result)

        if passed:
            logger.info(
                "DQ PASS | dataset=%s | check=%s | actual=%s | expected=%s",
                self.dataset_name,
                name,
                actual,
                expected,
            )
        else:
            logger.error(
                "DQ FAIL | dataset=%s | check=%s | actual=%s | expected=%s",
                self.dataset_name,
                name,
                actual,
                expected,
            )

    # ------------------------------------------------------------------
    # Row count
    # ------------------------------------------------------------------

    def check_row_count(
        self,
        minimum: int = 1,
    ) -> None:

        row_count = self.df.count()

        passed = row_count >= minimum

        self._record(
            name="row_count",
            passed=passed,
            actual=row_count,
            expected=f">= {minimum}",
            message=(f"Expected at least {minimum} rows, " f"found {row_count}"),
        )

    # ------------------------------------------------------------------
    # Null check
    # ------------------------------------------------------------------

    def check_not_null(
        self,
        column: str,
    ) -> None:

        null_count = self.df.filter(F.col(column).isNull()).count()

        passed = null_count == 0

        self._record(
            name=f"not_null:{column}",
            passed=passed,
            actual=null_count,
            expected="0",
            message=(f"Column '{column}' contains " f"{null_count} null values"),
        )

    # ------------------------------------------------------------------
    # Numeric minimum
    # ------------------------------------------------------------------

    def check_min_value(
        self,
        column: str,
        minimum: float,
    ) -> None:

        invalid_count = self.df.filter(F.col(column) < F.lit(minimum)).count()

        passed = invalid_count == 0

        self._record(
            name=f"minimum:{column}",
            passed=passed,
            actual=invalid_count,
            expected="0 invalid rows",
            message=(f"Column '{column}' has " f"{invalid_count} values below {minimum}"),
        )

    # ------------------------------------------------------------------
    # Duplicate key check
    # ------------------------------------------------------------------

    def check_unique(
        self,
        columns: list[str],
    ) -> None:

        duplicate_count = self.df.groupBy(*columns).count().filter(F.col("count") > 1).count()

        passed = duplicate_count == 0

        self._record(
            name=f"unique:{','.join(columns)}",
            passed=passed,
            actual=duplicate_count,
            expected="0",
            message=(f"Found {duplicate_count} duplicate " f"key groups for {columns}"),
        )

    # ------------------------------------------------------------------
    # Allowed schema
    # ------------------------------------------------------------------

    def check_columns(
        self,
        expected_columns: list[str],
    ) -> None:

        actual_columns = self.df.columns

        missing = [column for column in expected_columns if column not in actual_columns]

        passed = len(missing) == 0

        self._record(
            name="required_columns",
            passed=passed,
            actual=len(missing),
            expected="0 missing columns",
            message=(f"Missing columns: {missing}"),
        )

    # ------------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------------

    def validate(self) -> dict[str, object]:

        failed = [result for result in self.results if not result.passed]

        passed = len(failed) == 0

        logger.info(
            "DQ SUMMARY | dataset=%s | passed=%s | checks=%s | failures=%s",
            self.dataset_name,
            passed,
            len(self.results),
            len(failed),
        )

        if not passed:

            for result in failed:
                logger.error(
                    "DQ FAILURE | %s | %s",
                    result.name,
                    result.message,
                )

        return {
            "dataset": self.dataset_name,
            "passed": passed,
            "checks": len(self.results),
            "failures": len(failed),
            "results": self.results,
        }

    def validate_or_raise(self) -> None:

        summary = self.validate()

        if not summary["passed"]:

            failures = [result.message for result in self.results if not result.passed]

            raise RuntimeError(
                "Data quality validation failed for "
                f"{self.dataset_name}: " + " | ".join(failures)
            )
