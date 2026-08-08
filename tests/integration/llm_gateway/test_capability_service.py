import pytest

from ai_platform.llm_gateway.services.capability_service import (
    capability_service,
)


def test_openai_chat_model_supported():

    capability_service.validate_chat(
        "openai",
        "gpt-4.1",
    )


def test_openai_stream_model_supported():

    capability_service.validate_stream(
        "openai",
        "gpt-4.1",
    )


def test_gemini_stream_model_supported():

    capability_service.validate_stream(
        "gemini",
        "gemini-chat",
    )


def test_invalid_stream_model():

    with pytest.raises(ValueError) as excinfo:

        capability_service.validate_stream(
            "gemini",
            "does-not-exist",
        )

    assert "Unsupported gemini stream model" in str(excinfo.value)
