import importlib

import ai_platform.llm_gateway.config.provider_settings as provider_settings


def test_default_providers(monkeypatch):
    monkeypatch.delenv(
        "ENABLED_PROVIDERS",
        raising=False,
    )

    importlib.reload(provider_settings)

    assert provider_settings.get_enabled_providers() == provider_settings.DEFAULT_PROVIDERS


def test_custom_providers(monkeypatch):
    monkeypatch.setenv(
        "ENABLED_PROVIDERS",
        "openai,gemini",
    )

    importlib.reload(provider_settings)

    assert provider_settings.get_enabled_providers() == [
        "openai",
        "gemini",
    ]


def test_spaces_are_removed(monkeypatch):
    monkeypatch.setenv(
        "ENABLED_PROVIDERS",
        " openai , gemini , anthropic ",
    )

    importlib.reload(provider_settings)

    assert provider_settings.get_enabled_providers() == [
        "openai",
        "gemini",
        "anthropic",
    ]


def test_empty_entries_removed(monkeypatch):
    monkeypatch.setenv(
        "ENABLED_PROVIDERS",
        "openai,,gemini,,,",
    )

    importlib.reload(provider_settings)

    assert provider_settings.get_enabled_providers() == [
        "openai",
        "gemini",
    ]
