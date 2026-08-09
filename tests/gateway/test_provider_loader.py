from unittest.mock import MagicMock, patch, ANY

import pytest

from ai_platform.llm_gateway.providers.provider_loader import (
    load_providers,
)

###############################################################################
# Successful loading
###############################################################################


@patch(
    "ai_platform.llm_gateway.providers.provider_loader.PROVIDER_CLASSES",
    {
        "openai": MagicMock,
        "gemini": MagicMock,
    },
)
@patch(
    "ai_platform.llm_gateway.providers.provider_loader.get_enabled_providers",
)
def test_load_providers_success(mock_enabled):

    mock_enabled.return_value = [
        "openai",
        "gemini",
    ]

    registry = MagicMock()

    load_providers(registry)

    assert registry.register.call_count == 2

    registry.register.assert_any_call(
        "openai",
        ANY,
    )

    registry.register.assert_any_call(
        "gemini",
        ANY,
    )


###############################################################################
# No providers configured
###############################################################################


@patch(
    "ai_platform.llm_gateway.providers.provider_loader.get_enabled_providers",
)
def test_load_providers_none(mock_enabled):

    mock_enabled.return_value = []

    registry = MagicMock()

    with pytest.raises(ValueError, match="No providers configured"):
        load_providers(registry)


###############################################################################
# Unknown configured provider
###############################################################################


@patch(
    "ai_platform.llm_gateway.providers.provider_loader.get_enabled_providers",
)
def test_load_providers_unknown(mock_enabled):

    mock_enabled.return_value = [
        "bad-provider",
    ]

    registry = MagicMock()

    with pytest.raises(ValueError, match="Unknown configured provider"):
        load_providers(registry)
