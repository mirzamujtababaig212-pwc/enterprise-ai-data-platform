import pytest
from unittest.mock import patch

from ai_platform.llm_gateway.services.capability_service import (
    CapabilityService,
)
from ai_platform.llm_gateway.exceptions.gateway_exceptions import (
    ProviderNotFound,
)

###############################################################################
# provider_exists()
###############################################################################


def test_provider_exists_true():

    service = CapabilityService()

    with patch(
        "ai_platform.llm_gateway.services.capability_service.provider_exists",
        return_value=True,
    ):
        assert service.provider_exists("openai") is True


def test_provider_exists_false():

    service = CapabilityService()

    with patch(
        "ai_platform.llm_gateway.services.capability_service.provider_exists",
        return_value=False,
    ):
        assert service.provider_exists("unknown") is False


###############################################################################
# model_supported()
###############################################################################


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
                "gpt-4o",
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
                "bad-model",
            )
            is False
        )


###############################################################################
# validate_provider()
###############################################################################


def test_validate_provider_success():

    service = CapabilityService()

    with patch.object(
        service,
        "provider_exists",
        return_value=True,
    ):
        service.validate_provider("openai")


def test_validate_provider_failure():

    service = CapabilityService()

    with patch.object(
        service,
        "provider_exists",
        return_value=False,
    ):
        with pytest.raises(ProviderNotFound):
            service.validate_provider("bad-provider")


###############################################################################
# validate_chat()
###############################################################################


def test_validate_chat_success():

    service = CapabilityService()

    with (
        patch.object(service, "validate_provider"),
        patch.object(
            service,
            "model_supported",
            return_value=True,
        ),
    ):
        service.validate_chat(
            "openai",
            "gpt-4o",
        )


def test_validate_chat_failure():

    service = CapabilityService()

    with (
        patch.object(service, "validate_provider"),
        patch.object(
            service,
            "model_supported",
            return_value=False,
        ),
    ):
        with pytest.raises(ValueError):
            service.validate_chat(
                "openai",
                "bad-model",
            )


###############################################################################
# validate_embeddings()
###############################################################################


def test_validate_embeddings_success():

    service = CapabilityService()

    with (
        patch.object(service, "validate_provider"),
        patch.object(
            service,
            "model_supported",
            return_value=True,
        ),
    ):
        service.validate_embeddings(
            "openai",
            "text-embedding-3-small",
        )


def test_validate_embeddings_failure():

    service = CapabilityService()

    with (
        patch.object(service, "validate_provider"),
        patch.object(
            service,
            "model_supported",
            return_value=False,
        ),
    ):
        with pytest.raises(ValueError):
            service.validate_embeddings(
                "openai",
                "bad-model",
            )


###############################################################################
# validate_stream()
###############################################################################


def test_validate_stream_success():

    service = CapabilityService()

    with (
        patch.object(service, "validate_provider"),
        patch.object(
            service,
            "model_supported",
            return_value=True,
        ),
    ):
        service.validate_stream(
            "openai",
            "gpt-4o",
        )


def test_validate_stream_failure():

    service = CapabilityService()

    with (
        patch.object(service, "validate_provider"),
        patch.object(
            service,
            "model_supported",
            return_value=False,
        ),
    ):
        with pytest.raises(ValueError):
            service.validate_stream(
                "openai",
                "bad-model",
            )
