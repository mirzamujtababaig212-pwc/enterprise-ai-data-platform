import importlib
import pytest

import ai_platform.llm_gateway.config.openai_settings as openai_settings


def test_validate_openai_settings_success(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    importlib.reload(openai_settings)

    openai_settings.validate_openai_settings()


def test_validate_openai_settings_failure(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    importlib.reload(openai_settings)

    with pytest.raises(ValueError, match="OPENAI_API_KEY is missing."):
        openai_settings.validate_openai_settings()


def test_default_model(monkeypatch):
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    importlib.reload(openai_settings)

    assert openai_settings.OPENAI_MODEL == "gpt-4.1-mini"


def test_custom_model(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")

    importlib.reload(openai_settings)

    assert openai_settings.OPENAI_MODEL == "gpt-4o"


def test_default_base_url(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    importlib.reload(openai_settings)

    assert openai_settings.OPENAI_BASE_URL == "https://api.openai.com/v1"


def test_custom_base_url(monkeypatch):
    monkeypatch.setenv(
        "OPENAI_BASE_URL",
        "https://example.com/v1",
    )

    importlib.reload(openai_settings)

    assert openai_settings.OPENAI_BASE_URL == "https://example.com/v1"
