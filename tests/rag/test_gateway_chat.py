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


@pytest.mark.asyncio
async def test_gateway_chat_service_passes_structured_output_to_llm_gateway():
    fake_router = MagicMock()

    fake_router.route_chat = AsyncMock(
        return_value={
            "reply": '{"answer":"hello"}',
        }
    )

    service = GatewayChatService(
        provider="openai",
        model="gpt-4o",
        gateway_router=fake_router,
    )

    structured_output = {
        "name": "test_result",
        "schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
            },
            "required": ["answer"],
            "additionalProperties": False,
        },
        "strict": True,
    }

    response = await service.generate(
        "Return a result",
        structured_output=structured_output,
    )

    assert response["reply"] == '{"answer":"hello"}'

    fake_router.route_chat.assert_awaited_once_with(
        {
            "provider": "openai",
            "model": "gpt-4o",
            "prompt": "Return a result",
            "temperature": 0.2,
            "max_tokens": 1024,
            "stream": False,
            "structured_output": structured_output,
        }
    )
