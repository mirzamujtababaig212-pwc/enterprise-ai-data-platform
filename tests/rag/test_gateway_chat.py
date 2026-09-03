from unittest.mock import AsyncMock, MagicMock

import pytest

from rag.generation.gateway import GatewayChatService


@pytest.mark.asyncio
async def test_gateway_chat_service_uses_llm_gateway():

    fake_router = MagicMock()

    fake_router.route_chat = AsyncMock(
        return_value={
            "reply": "hello",
        }
    )

    service = GatewayChatService(
        provider="mock",
        model="mock-gpt",
        gateway_router=fake_router,
    )

    response = await service.generate(
        "Hello",
    )

    assert response["reply"] == "hello"

    fake_router.route_chat.assert_awaited_once_with(
        {
            "provider": "mock",
            "model": "mock-gpt",
            "prompt": "Hello",
            "temperature": 0.2,
            "max_tokens": 1024,
            "stream": False,
        }
    )
