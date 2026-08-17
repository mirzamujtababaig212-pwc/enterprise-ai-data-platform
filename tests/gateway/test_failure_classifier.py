"""Tests for provider failure classification."""

from ai_platform.llm_gateway.reliability.failure_classifier import (
    FailureCategory,
    ProviderFailureClassifier,
)


def test_timeout_is_retryable() -> None:
    classifier = ProviderFailureClassifier()

    error = TimeoutError("provider request timeout")

    assert classifier.classify(error) == FailureCategory.TIMEOUT
    assert classifier.is_retryable(error)
    assert classifier.is_fallback_eligible(error)


def test_rate_limit_is_retryable() -> None:
    classifier = ProviderFailureClassifier()

    error = RuntimeError("429 too_many_requests")

    assert classifier.classify(error) == FailureCategory.RATE_LIMITED
    assert classifier.is_retryable(error)
    assert classifier.is_fallback_eligible(error)


def test_connection_error_is_transient() -> None:
    classifier = ProviderFailureClassifier()

    error = ConnectionError("provider unavailable")

    assert classifier.classify(error) == FailureCategory.TRANSIENT
    assert classifier.is_retryable(error)
    assert classifier.is_fallback_eligible(error)


def test_authentication_failure_is_not_retryable() -> None:
    classifier = ProviderFailureClassifier()

    error = RuntimeError("authentication failed")

    assert classifier.classify(error) == FailureCategory.AUTHENTICATION
    assert not classifier.is_retryable(error)
    assert not classifier.is_fallback_eligible(error)


def test_authorization_failure_is_not_retryable() -> None:
    classifier = ProviderFailureClassifier()

    error = RuntimeError("403 forbidden")

    assert classifier.classify(error) == FailureCategory.AUTHORIZATION
    assert not classifier.is_retryable(error)
    assert not classifier.is_fallback_eligible(error)


def test_invalid_request_is_not_retryable() -> None:
    classifier = ProviderFailureClassifier()

    error = RuntimeError("invalid_request")

    assert classifier.classify(error) == FailureCategory.INVALID_REQUEST
    assert not classifier.is_retryable(error)
    assert not classifier.is_fallback_eligible(error)


def test_model_not_found_is_not_retryable() -> None:
    classifier = ProviderFailureClassifier()

    error = RuntimeError("model_not_found")

    assert classifier.classify(error) == FailureCategory.NOT_FOUND
    assert not classifier.is_retryable(error)
    assert not classifier.is_fallback_eligible(error)


def test_unknown_error_can_fallback() -> None:
    classifier = ProviderFailureClassifier()

    error = RuntimeError("unexpected provider failure")

    assert classifier.classify(error) == FailureCategory.UNKNOWN
    assert not classifier.is_retryable(error)
    assert classifier.is_fallback_eligible(error)
