import asyncio

import pytest
from pydantic import ValidationError

from app.control_plane.retries.classifier import FailureClassifier
from app.control_plane.retries.models import (
    FailureClassification,
    FailureDisposition,
)
from app.control_plane.retries.policy import RetryPolicy


def test_default_policy_validation() -> None:
    policy = RetryPolicy()

    assert policy.max_attempts == 3
    assert policy.initial_backoff_seconds == 1.0
    assert policy.max_backoff_seconds == 30.0
    assert policy.jitter == 0.1
    assert "timeout" in policy.retryable_categories


def test_invalid_policy_bounds() -> None:
    with pytest.raises(ValidationError):
        RetryPolicy(max_attempts=0)

    with pytest.raises(ValidationError):
        RetryPolicy(
            initial_backoff_seconds=10.0,
            max_backoff_seconds=5.0,
        )


def test_retryable_category_matching() -> None:
    policy = RetryPolicy()

    decision = policy.evaluate(
        category="timeout",
        disposition=FailureDisposition.RETRYABLE,
        attempt=1,
    )

    assert decision.allowed is True
    assert decision.attempt == 1


def test_non_retryable_category_rejection() -> None:
    policy = RetryPolicy()

    decision = policy.evaluate(
        category="validation_error",
        disposition=FailureDisposition.NON_RETRYABLE,
        attempt=1,
    )

    assert decision.allowed is False
    assert "non-retryable" in decision.reason


def test_unconfigured_category_rejection() -> None:
    policy = RetryPolicy(retryable_categories={"timeout"})

    decision = policy.evaluate(
        category="rate_limit_exceeded",
        disposition=FailureDisposition.RETRYABLE,
        attempt=1,
    )

    assert decision.allowed is False
    assert "not configured as retryable" in decision.reason


def test_attempt_boundary() -> None:
    policy = RetryPolicy(max_attempts=3)

    assert (
        policy.evaluate(
            "timeout",
            FailureDisposition.RETRYABLE,
            attempt=1,
        ).allowed
        is True
    )

    assert (
        policy.evaluate(
            "timeout",
            FailureDisposition.RETRYABLE,
            attempt=2,
        ).allowed
        is True
    )

    decision = policy.evaluate(
        "timeout",
        FailureDisposition.RETRYABLE,
        attempt=3,
    )

    assert decision.allowed is False
    assert "Exceeded max attempts" in decision.reason


def test_invalid_attempt_is_rejected() -> None:
    policy = RetryPolicy()

    with pytest.raises(ValueError):
        policy.calculate_delay(attempt=0)

    with pytest.raises(ValueError):
        policy.evaluate(
            "timeout",
            FailureDisposition.RETRYABLE,
            attempt=0,
        )


def test_exponential_backoff_and_max_cap() -> None:
    policy = RetryPolicy(
        initial_backoff_seconds=1.0,
        backoff_factor=2.0,
        max_backoff_seconds=5.0,
        jitter=0.0,
    )

    assert policy.calculate_delay(attempt=1) == 1.0
    assert policy.calculate_delay(attempt=2) == 2.0
    assert policy.calculate_delay(attempt=3) == 4.0
    assert policy.calculate_delay(attempt=4) == 5.0


def test_deterministic_and_bounded_jitter() -> None:
    policy = RetryPolicy(
        initial_backoff_seconds=10.0,
        jitter=0.2,
    )

    assert (
        policy.calculate_delay(
            attempt=1,
            rng=1.0,
        )
        == 12.0
    )

    assert (
        policy.calculate_delay(
            attempt=1,
            rng=-1.0,
        )
        == 8.0
    )


def test_jitter_respects_absolute_max_backoff() -> None:
    policy = RetryPolicy(
        initial_backoff_seconds=20.0,
        backoff_factor=2.0,
        max_backoff_seconds=30.0,
        jitter=0.5,
    )

    assert (
        policy.calculate_delay(
            attempt=2,
            rng=1.0,
        )
        == 30.0
    )


def test_invalid_jitter_random_factor_is_rejected() -> None:
    policy = RetryPolicy(jitter=0.2)

    with pytest.raises(ValueError):
        policy.calculate_delay(attempt=1, rng=2.0)

    with pytest.raises(ValueError):
        policy.calculate_delay(attempt=1, rng=-2.0)


def test_zero_backoff_behavior() -> None:
    policy = RetryPolicy(
        initial_backoff_seconds=0.0,
        jitter=0.5,
    )

    assert policy.calculate_delay(attempt=1) == 0.0


def test_known_exception_classification() -> None:
    classifier = FailureClassifier()

    timeout_result = classifier.classify(TimeoutError("request timed out"))
    assert timeout_result.category == "timeout"
    assert timeout_result.disposition == FailureDisposition.RETRYABLE

    connection_result = classifier.classify(ConnectionResetError("connection lost"))
    assert connection_result.category == "connection_error"
    assert connection_result.disposition == FailureDisposition.RETRYABLE


def test_async_timeout_classification() -> None:
    classifier = FailureClassifier()

    result = classifier.classify(asyncio.TimeoutError("request timed out"))

    assert result.category == "timeout"
    assert result.disposition == FailureDisposition.RETRYABLE


def test_unknown_exception_classification() -> None:
    classifier = FailureClassifier()

    class CustomDomainException(Exception):
        pass

    result = classifier.classify(CustomDomainException("unexpected issue"))

    assert result.category == "unknown"
    assert result.disposition == FailureDisposition.NON_RETRYABLE


def test_approval_rejected_never_classified_as_retryable() -> None:
    classifier = FailureClassifier()

    class ApprovalRejectedError(Exception):
        pass

    result = classifier.classify(ApprovalRejectedError("User declined execution"))

    assert result.category == "approval_rejected"
    assert result.disposition == FailureDisposition.NON_RETRYABLE


def test_ambiguous_execution_remains_ambiguous() -> None:
    classifier = FailureClassifier()
    policy = RetryPolicy()

    exc = RuntimeError("Execution status is ambiguous_execution following lease loss")
    result = classifier.classify(exc)

    assert result.disposition == FailureDisposition.AMBIGUOUS

    decision = policy.evaluate(
        result.category,
        result.disposition,
        attempt=1,
    )

    assert decision.allowed is False
    assert "Ambiguous failure" in decision.reason


def test_os_error_is_not_automatically_retryable() -> None:
    classifier = FailureClassifier()

    result = classifier.classify(PermissionError("permission denied"))

    assert result.category == "unknown"
    assert result.disposition == FailureDisposition.NON_RETRYABLE


def test_policy_and_classification_serialization() -> None:
    classification = FailureClassification(
        category="rate_limit_exceeded",
        disposition=FailureDisposition.RETRYABLE,
        reason="Rate limit triggered",
    )

    dumped = classification.model_dump()

    assert dumped["disposition"] == "retryable"

    policy = RetryPolicy()
    policy_dumped = policy.model_dump()

    assert "timeout" in policy_dumped["retryable_categories"]
