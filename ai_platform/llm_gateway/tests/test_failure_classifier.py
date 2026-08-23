from ai_platform.llm_gateway.reliability.failure_classifier import (
    FailureCategory,
    failure_classifier,
)


def test_insufficient_quota_is_not_retryable():
    error = Exception("OpenAI error code: insufficient_quota")

    assert failure_classifier.classify(error) == FailureCategory.QUOTA_EXCEEDED
    assert failure_classifier.is_retryable(error) is False
    assert failure_classifier.is_fallback_eligible(error) is True


def test_rate_limit_is_retryable():
    error = Exception("OpenAI rate_limit exceeded")

    assert failure_classifier.classify(error) == FailureCategory.RATE_LIMITED
    assert failure_classifier.is_retryable(error) is True
    assert failure_classifier.is_fallback_eligible(error) is True
