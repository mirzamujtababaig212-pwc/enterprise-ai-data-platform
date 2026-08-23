from ai_platform.llm_gateway.reliability.retry import should_retry


def test_429_is_retryable():
    result = should_retry(
        status_code=429,
        attempt=0,
        max_attempts=3,
        base_delay=1.0,
        jitter=False,
    )

    assert result.retry is True
    assert result.delay_seconds == 1.0


def test_500_is_retryable():
    result = should_retry(
        status_code=500,
        attempt=1,
        max_attempts=3,
        base_delay=1.0,
        jitter=False,
    )

    assert result.retry is True
    assert result.delay_seconds == 2.0


def test_retry_delay_is_capped():
    result = should_retry(
        status_code=503,
        attempt=4,
        max_attempts=6,
        base_delay=1.0,
        max_delay=8.0,
        jitter=False,
    )

    assert result.retry is True
    assert result.delay_seconds == 8.0


def test_non_retryable_status_does_not_retry():
    result = should_retry(
        status_code=401,
        attempt=0,
        max_attempts=3,
        jitter=False,
    )

    assert result.retry is False
    assert result.delay_seconds == 0.0


def test_last_attempt_does_not_retry():
    result = should_retry(
        status_code=500,
        attempt=2,
        max_attempts=3,
        jitter=False,
    )

    assert result.retry is False
    assert result.delay_seconds == 0.0
