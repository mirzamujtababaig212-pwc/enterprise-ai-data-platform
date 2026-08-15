"""Failure classification for provider routing and fallback."""

from __future__ import annotations

from enum import Enum
from typing import Any


class FailureCategory(str, Enum):
    """Categories used to determine whether a provider failure is retryable."""

    TRANSIENT = "transient"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    INVALID_REQUEST = "invalid_request"
    NOT_FOUND = "not_found"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


class ProviderFailureClassifier:
    """Classify provider exceptions for retry/fallback decisions."""

    _RATE_LIMIT_NAMES = (
        "ratelimit",
        "rate_limit",
        "too_many_requests",
    )

    _TIMEOUT_NAMES = (
        "timeout",
        "timedout",
    )

    _AUTH_NAMES = (
        "authentication",
        "unauthorized",
        "invalid_api_key",
        "credentials",
    )

    _AUTHORIZATION_NAMES = (
        "forbidden",
        "authorization",
        "permission",
    )

    _INVALID_REQUEST_NAMES = (
        "invalid_request",
        "bad_request",
        "validation",
    )

    _NOT_FOUND_NAMES = (
        "not_found",
        "does_not_exist",
        "model_not_found",
    )

    _TRANSIENT_NAMES = (
        "connection",
        "connect",
        "unavailable",
        "service_unavailable",
        "temporarily",
        "network",
        "transport",
        "overloaded",
    )

    def classify(self, error: BaseException) -> FailureCategory:
        """Return the best matching failure category."""

        text = self._exception_text(error)

        if self._contains_any(text, self._RATE_LIMIT_NAMES):
            return FailureCategory.RATE_LIMITED

        if self._contains_any(text, self._TIMEOUT_NAMES):
            return FailureCategory.TIMEOUT

        if self._contains_any(text, self._AUTH_NAMES):
            return FailureCategory.AUTHENTICATION

        if self._contains_any(text, self._AUTHORIZATION_NAMES):
            return FailureCategory.AUTHORIZATION

        if self._contains_any(text, self._INVALID_REQUEST_NAMES):
            return FailureCategory.INVALID_REQUEST

        if self._contains_any(text, self._NOT_FOUND_NAMES):
            return FailureCategory.NOT_FOUND

        if self._contains_any(text, self._TRANSIENT_NAMES):
            return FailureCategory.TRANSIENT

        return FailureCategory.UNKNOWN

    def is_retryable(self, error: BaseException) -> bool:
        """Return whether retry/fallback is appropriate."""

        category = self.classify(error)

        return category in {
            FailureCategory.TRANSIENT,
            FailureCategory.RATE_LIMITED,
            FailureCategory.TIMEOUT,
        }

    def is_fallback_eligible(self, error: BaseException) -> bool:
        """Return whether another provider should be attempted."""

        category = self.classify(error)

        return category in {
            FailureCategory.TRANSIENT,
            FailureCategory.RATE_LIMITED,
            FailureCategory.TIMEOUT,
            FailureCategory.UNKNOWN,
        }

    @staticmethod
    def _exception_text(error: BaseException) -> str:
        """Build a searchable representation of an exception."""

        parts: list[str] = [
            error.__class__.__name__,
            str(error),
        ]

        for attribute in (
            "status_code",
            "status",
            "code",
            "message",
        ):
            value: Any = getattr(error, attribute, None)

            if value is not None:
                parts.append(str(value))

        return " ".join(parts).lower()

    @staticmethod
    def _contains_any(
        text: str,
        patterns: tuple[str, ...],
    ) -> bool:
        """Return True when any pattern appears in text."""

        return any(pattern in text for pattern in patterns)


failure_classifier = ProviderFailureClassifier()
