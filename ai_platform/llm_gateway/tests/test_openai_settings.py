from ai_platform.llm_gateway.config.openai_settings import (
    get_openai_settings,
)
from ai_platform.llm_gateway.reliability.failure_classifier import (
    FailureCategory,
    failure_classifier,
)


def test_direct_environment_variables_take_precedence(monkeypatch):
    monkeypatch.setenv(
        "OPENAI_API_KEY",
        "direct-test-key",
    )
    monkeypatch.setenv(
        "OPENAI_MODEL",
        "gpt-4.1",
    )
    monkeypatch.setenv(
        "OPENAI_BASE_URL",
        "https://example.openai.test/v1",
    )
    monkeypatch.setenv(
        "PROVIDER_CREDENTIALS",
        '{"openai":{"api_key":"fallback-key","model":"gpt-4o-mini"}}',
    )

    settings = get_openai_settings()

    assert settings.api_key == "direct-test-key"
    assert settings.model == "gpt-4.1"
    assert settings.base_url == "https://example.openai.test/v1"


def test_provider_credentials_are_used_as_fallback(monkeypatch):
    monkeypatch.delenv(
        "OPENAI_API_KEY",
        raising=False,
    )
    monkeypatch.delenv(
        "OPENAI_MODEL",
        raising=False,
    )
    monkeypatch.delenv(
        "OPENAI_BASE_URL",
        raising=False,
    )

    monkeypatch.setenv(
        "PROVIDER_CREDENTIALS",
        '{"openai":{"api_key":"fallback-key","model":"gpt-4o-mini"}}',
    )

    settings = get_openai_settings()

    assert settings.api_key == "fallback-key"
    assert settings.model == "gpt-4o-mini"


def test_default_settings(monkeypatch):
    monkeypatch.delenv(
        "OPENAI_API_KEY",
        raising=False,
    )
    monkeypatch.delenv(
        "OPENAI_MODEL",
        raising=False,
    )
    monkeypatch.delenv(
        "OPENAI_BASE_URL",
        raising=False,
    )
    monkeypatch.delenv(
        "PROVIDER_CREDENTIALS",
        raising=False,
    )

    settings = get_openai_settings()

    assert settings.api_key == ""
    assert settings.model == "gpt-4.1-mini"
    assert settings.base_url == "https://api.openai.com/v1"
    assert settings.timeout == 60.0
    assert settings.max_retries == 0


def test_invalid_provider_credentials_do_not_crash(monkeypatch):
    monkeypatch.delenv(
        "OPENAI_API_KEY",
        raising=False,
    )

    monkeypatch.setenv(
        "PROVIDER_CREDENTIALS",
        "not-valid-json",
    )

    settings = get_openai_settings()

    assert settings.api_key == ""


def test_insufficient_quota_is_not_retryable():
    error = Exception("OpenAI error code: insufficient_quota")

    assert failure_classifier.classify(error) == (FailureCategory.QUOTA_EXCEEDED)

    assert failure_classifier.is_retryable(error) is False
    assert failure_classifier.is_fallback_eligible(error) is True


def test_rate_limit_is_retryable():
    error = Exception("OpenAI rate_limit exceeded")

    assert failure_classifier.classify(error) == (FailureCategory.RATE_LIMITED)

    assert failure_classifier.is_retryable(error) is True
    assert failure_classifier.is_fallback_eligible(error) is True
