from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FailureClassification:
    category: str
    retryable: bool
    reason: str


class FailureClassifier:
    """
    Classify pipeline failures into operational categories.

    This classifier is intentionally conservative.

    Unknown errors are NOT automatically retryable.
    """

    RETRYABLE_PATTERNS = (
        "timeout",
        "timed out",
        "connection reset",
        "connection refused",
        "temporarily unavailable",
        "service unavailable",
        "throttl",
        "rate exceeded",
        "too many requests",
        "network error",
        "socket timeout",
        "executor lost",
        "fetch failed",
        "temporary failure",
    )

    NON_RETRYABLE_PATTERNS = (
        "syntax error",
        "analysisexception",
        "column not found",
        "table or view not found",
        "database not found",
        "schema not found",
        "permission denied",
        "access denied",
        "invalid configuration",
        "cannot resolve",
        "parseexception",
        "datatype mismatch",
    )

    @classmethod
    def classify(
        cls,
        error_message: str | None,
    ) -> FailureClassification:

        if not error_message:
            return FailureClassification(
                category="UNKNOWN",
                retryable=False,
                reason="No error message was provided.",
            )

        message = error_message.lower()

        for pattern in cls.RETRYABLE_PATTERNS:
            if pattern in message:
                return FailureClassification(
                    category="TRANSIENT",
                    retryable=True,
                    reason=(f"Matched retryable pattern: " f"{pattern}"),
                )

        for pattern in cls.NON_RETRYABLE_PATTERNS:
            if pattern in message:
                return FailureClassification(
                    category="PERSISTENT",
                    retryable=False,
                    reason=(f"Matched non-retryable pattern: " f"{pattern}"),
                )

        return FailureClassification(
            category="UNKNOWN",
            retryable=False,
            reason=("No known failure signature matched. " "Retry is disabled by default."),
        )
