from ai_platform.llm_gateway.reliability.retry import should_retry


def test_429_is_retryable():
    result = should_retry(
        status_code=429,
        attempt=0,
        max_attempts=3,
    )

    assert result.retry is True
    assert result.delay_seconds == 1


def test_500_is_retryable():
    result = should_retry(
        status_code=500,
        attempt=1,
        max_attempts=3,
    )

    assert result.retry is True
    assert result.delay_seconds == 2


def test_400_is_not_retryable():
    result = should_retry(
        status_code=400,
        attempt=0,
        max_attempts=3,
    )

    assert result.retry is False


def test_max_attempts_prevents_retry():
    result = should_retry(
        status_code=429,
        attempt=3,
        max_attempts=3,
    )

    assert result.retry is False
