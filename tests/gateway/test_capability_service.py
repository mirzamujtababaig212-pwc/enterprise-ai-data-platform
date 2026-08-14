from unittest.mock import patch

import pytest

from ai_platform.llm_gateway.exceptions.gateway_exceptions import (
    ProviderNotFound,
)
from ai_platform.llm_gateway.services.capability_service import (
    CapabilityService,
)


def test_provider_exists_true():
    service = CapabilityService()

    with patch(
        "ai_platform.llm_gateway.services.capability_service.provider_exists",
        return_value=True,
    ):
        assert (
            service.provider_exists(
                "openai",
            )
            is True
        )


def test_provider_exists_false():
    service = CapabilityService()

    with patch(
        "ai_platform.llm_gateway.services.capability_service.provider_exists",
        return_value=False,
    ):
        assert (
            service.provider_exists(
                "unknown",
            )
            is False
        )


def test_model_supported_true():
    service = CapabilityService()

    with patch(
        "ai_platform.llm_gateway.services.capability_service.model_supported",
        return_value=True,
    ):
        assert (
            service.model_supported(
                "openai",
                "chat",
                "gpt-4.1",
            )
            is True
        )


def test_model_supported_false():
    service = CapabilityService()

    with patch(
        "ai_platform.llm_gateway.services.capability_service.model_supported",
        return_value=False,
    ):
        assert (
            service.model_supported(
                "openai",
                "chat",
                "invalid-model",
            )
            is False
        )


def test_validate_provider_success():
    service = CapabilityService()

    with patch.object(
        service,
        "provider_exists",
        return_value=True,
    ):
        service.validate_provider(
            "openai",
        )


def test_validate_provider_failure():
    service = CapabilityService()

    with patch.object(
        service,
        "provider_exists",
        return_value=False,
    ):
        with pytest.raises(
            ProviderNotFound,
            match="Unsupported provider",
        ):
            service.validate_provider(
                "unknown",
            )


def test_validate_model_success():
    service = CapabilityService()

    with (
        patch.object(
            service,
            "validate_provider",
        ),
        patch.object(
            service,
            "model_supported",
            return_value=True,
        ),
    ):
        service.validate_model(
            "openai",
            "chat",
            "gpt-4.1",
        )


def test_validate_model_failure():
    service = CapabilityService()

    with (
        patch.object(
            service,
            "validate_provider",
        ),
        patch.object(
            service,
            "model_supported",
            return_value=False,
        ),
    ):
        with pytest.raises(
            ValueError,
            match="Unsupported openai chat model",
        ):
            service.validate_model(
                "openai",
                "chat",
                "invalid-model",
            )


def test_validate_chat_success():
    service = CapabilityService()

    with patch.object(
        service,
        "validate_model",
    ) as validate_model:
        service.validate_chat(
            "openai",
            "gpt-4.1",
        )

    validate_model.assert_called_once_with(
        "openai",
        "chat",
        "gpt-4.1",
    )


def test_validate_embeddings_success():
    service = CapabilityService()

    with patch.object(
        service,
        "validate_model",
    ) as validate_model:
        service.validate_embeddings(
            "openai",
            "openai-embedding",
        )

    validate_model.assert_called_once_with(
        "openai",
        "embeddings",
        "openai-embedding",
    )


def test_validate_stream_success():
    service = CapabilityService()

    with patch.object(
        service,
        "validate_model",
    ) as validate_model:
        service.validate_stream(
            "openai",
            "gpt-4.1",
        )

    validate_model.assert_called_once_with(
        "openai",
        "stream",
        "gpt-4.1",
    )
