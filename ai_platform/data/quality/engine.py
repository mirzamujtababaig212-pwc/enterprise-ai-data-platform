"""Data quality rule orchestration."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from ai_platform.data.contracts import DataQualityRule

Record = dict[str, Any]


@dataclass(frozen=True)
class QualityReport:
    """Aggregate result for a collection of quality rules."""

    overall_passed: bool
    total_rules: int
    passed_rules: int
    failed_rules: int
    results: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable quality report."""

        return {
            "overall_passed": self.overall_passed,
            "total_rules": self.total_rules,
            "passed_rules": self.passed_rules,
            "failed_rules": self.failed_rules,
            "results": list(self.results),
        }


class QualityEngine:
    """Execute multiple data quality rules against the same dataset."""

    def __init__(
        self,
        rules: Iterable[DataQualityRule],
    ) -> None:
        self.rules = list(rules)

        for rule in self.rules:
            if not isinstance(rule, DataQualityRule):
                raise TypeError(
                    "QualityEngine rules must implement " "the DataQualityRule protocol."
                )

    def evaluate(
        self,
        data: Iterable[Record],
    ) -> dict[str, Any]:
        """Evaluate all configured rules."""

        records = list(data)

        results = [rule.evaluate(records) for rule in self.rules]

        passed_rules = sum(1 for result in results if result.get("passed") is True)

        failed_rules = len(results) - passed_rules

        report = QualityReport(
            overall_passed=failed_rules == 0,
            total_rules=len(results),
            passed_rules=passed_rules,
            failed_rules=failed_rules,
            results=results,
        )

        return report.to_dict()
