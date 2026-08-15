from unittest.mock import AsyncMock, Mock

import pytest

from ai_platform.llm_gateway.routing.fallback_executor import FallbackExecutor
from ai_platform.llm_gateway.routing.router import Router


class FakeProvider:
    def __init__(
        self,
        name: str,
        chat_result=None,
        chat_exception: Exception | None = None,
        embeddings_result=None,
        embeddings_exception: Exception | None = None,
        stream_chunks=None,
        stream_exception: Exception | None = None,
    ):
        self.name = name
        self.chat_result = (
            chat_result
            if chat_result is not None
            else {
                "provider": name,
                "response": f"{name}-chat-response",
            }
        )
        self.chat_exception = chat_exception

        self.embeddings_result = (
            embeddings_result if embeddings_result is not None else [0.1, 0.2, 0.3]
        )
        self.embeddings_exception = embeddings_exception

        self.stream_chunks = (
            stream_chunks if stream_chunks is not None else [f"{name}-chunk-1", f"{name}-chunk-2"]
        )
        self.stream_exception = stream_exception

        self.chat = AsyncMock(side_effect=self._chat)
        self.embeddings = AsyncMock(side_effect=self._embeddings)

    async def _chat(self, request):
        if self.chat_exception is not None:
            raise self.chat_exception

        return self.chat_result

    async def _embeddings(self, request):
        if self.embeddings_exception is not None:
            raise self.embeddings_exception

        return self.embeddings_result

    async def stream(self, request):
        if self.stream_exception is not None:
            raise self.stream_exception

        for chunk in self.stream_chunks:
            yield chunk


class FakeRoutingResolver:
    def __init__(
        self,
        providers,
        names=None,
    ):
        self.providers = providers
        self.names = names or [provider.name for provider in providers]

        self.resolve = Mock(return_value=list(self.providers))
        self.resolve_names = Mock(return_value=list(self.names))

    def resolve(
        self,
        capability,
        model,
        requested_provider=None,
    ):
        return list(self.providers)


class FakeCapabilityService:
    def validate_chat(self, provider, model):
        return None

    def validate_embeddings(self, provider, model):
        return None

    def validate_stream(self, provider, model):
        return None


class RecordingFallbackExecutor:
    def __init__(self, result=None, exception=None):
        self.result = result
        self.exception = exception
        self.execute_calls = []

    async def execute(self, providers, operation):
        self.execute_calls.append(
            {
                "providers": providers,
                "operation": operation,
            }
        )

        if self.exception is not None:
            raise self.exception

        return self.result


@pytest.fixture
def capability_service(monkeypatch):
    service = FakeCapabilityService()

    monkeypatch.setattr(
        "ai_platform.llm_gateway.routing.router.capability_service",
        service,
    )

    return service


def build_router(
    providers,
    fallback_executor=None,
):
    routing_resolver = FakeRoutingResolver(providers)

    router = Router(
        routing_resolver=routing_resolver,
        fallback_executor=fallback_executor or FallbackExecutor(),
    )

    return router, routing_resolver


@pytest.mark.asyncio
async def test_chat_first_provider_succeeds_second_is_never_called(
    capability_service,
):
    provider_a = FakeProvider(
        "provider-a",
        chat_result={
            "provider": "provider-a",
            "response": "success-a",
        },
    )

    provider_b = FakeProvider(
        "provider-b",
        chat_result={
            "provider": "provider-b",
            "response": "success-b",
        },
    )

    fallback_executor = FallbackExecutor()

    router, routing_resolver = build_router(
        [provider_a, provider_b],
        fallback_executor=fallback_executor,
    )

    request = {
        "model": "test-model",
        "messages": [
            {
                "role": "user",
                "content": "hello",
            }
        ],
    }

    result = await router.route_chat(request)

    assert result["response"] == "success-a"

    provider_a.chat.assert_awaited_once()
    provider_b.chat.assert_not_awaited()

    routing_resolver.resolve.assert_called_once_with(
        capability="chat",
        model="test-model",
        requested_provider=None,
    )


@pytest.mark.asyncio
async def test_chat_first_provider_fails_second_succeeds(
    capability_service,
):
    provider_a = FakeProvider(
        "provider-a",
        chat_exception=RuntimeError("provider-a failure"),
    )

    provider_b = FakeProvider(
        "provider-b",
        chat_result={
            "provider": "provider-b",
            "response": "success-b",
        },
    )

    fallback_executor = FallbackExecutor()

    router, routing_resolver = build_router(
        [provider_a, provider_b],
        fallback_executor=fallback_executor,
    )

    request = {
        "model": "test-model",
        "messages": [
            {
                "role": "user",
                "content": "hello",
            }
        ],
    }

    result = await router.route_chat(request)

    assert result["response"] == "success-b"

    provider_a.chat.assert_awaited_once()
    provider_b.chat.assert_awaited_once()


@pytest.mark.asyncio
async def test_chat_first_and_second_fail_third_succeeds(
    capability_service,
):
    provider_a = FakeProvider(
        "provider-a",
        chat_exception=RuntimeError("provider-a failure"),
    )

    provider_b = FakeProvider(
        "provider-b",
        chat_exception=RuntimeError("provider-b failure"),
    )

    provider_c = FakeProvider(
        "provider-c",
        chat_result={
            "provider": "provider-c",
            "response": "success-c",
        },
    )

    fallback_executor = FallbackExecutor()

    router, routing_resolver = build_router(
        [provider_a, provider_b, provider_c],
        fallback_executor=fallback_executor,
    )

    request = {
        "model": "test-model",
        "messages": [
            {
                "role": "user",
                "content": "hello",
            }
        ],
    }

    result = await router.route_chat(request)

    assert result["response"] == "success-c"

    provider_a.chat.assert_awaited_once()
    provider_b.chat.assert_awaited_once()
    provider_c.chat.assert_awaited_once()


@pytest.mark.asyncio
async def test_chat_all_providers_fail_raises_final_classified_exception(
    capability_service,
):
    provider_a = FakeProvider(
        "provider-a",
        chat_exception=RuntimeError("provider-a failure"),
    )

    provider_b = FakeProvider(
        "provider-b",
        chat_exception=RuntimeError("provider-b failure"),
    )

    provider_c = FakeProvider(
        "provider-c",
        chat_exception=RuntimeError("provider-c failure"),
    )

    fallback_executor = FallbackExecutor()

    router, routing_resolver = build_router(
        [provider_a, provider_b, provider_c],
        fallback_executor=fallback_executor,
    )

    request = {
        "model": "test-model",
        "messages": [
            {
                "role": "user",
                "content": "hello",
            }
        ],
    }

    with pytest.raises(RuntimeError, match="provider-c failure") as exc_info:
        await router.route_chat(request)

    assert exc_info.value is not None

    provider_a.chat.assert_awaited_once()
    provider_b.chat.assert_awaited_once()
    provider_c.chat.assert_awaited_once()


@pytest.mark.asyncio
async def test_explicit_provider_is_preserved(
    capability_service,
):
    provider_a = FakeProvider(
        "provider-a",
        chat_result={
            "provider": "provider-a",
            "response": "provider-a-response",
        },
    )

    provider_b = FakeProvider(
        "provider-b",
        chat_result={
            "provider": "provider-b",
            "response": "provider-b-response",
        },
    )

    fallback_executor = FallbackExecutor()

    router, routing_resolver = build_router(
        [provider_a, provider_b],
        fallback_executor=fallback_executor,
    )

    request = {
        "provider": "mock",
        "model": "test-model",
        "messages": [
            {
                "role": "user",
                "content": "hello",
            }
        ],
    }

    result = await router.route_chat(request)

    assert result["response"] == "provider-a-response"

    provider_a.chat.assert_awaited_once()
    provider_b.chat.assert_not_awaited()

    routing_resolver.resolve.assert_called_once_with(
        capability="chat",
        model="test-model",
        requested_provider="mock",
    )


@pytest.mark.asyncio
async def test_embeddings_use_fallback_chain(
    capability_service,
):
    provider_a = FakeProvider(
        "provider-a",
        embeddings_exception=RuntimeError("provider-a embeddings failure"),
    )

    provider_b = FakeProvider(
        "provider-b",
        embeddings_result=[0.9, 0.8, 0.7],
    )

    fallback_executor = FallbackExecutor()

    router, routing_resolver = build_router(
        [provider_a, provider_b],
        fallback_executor=fallback_executor,
    )

    request = {
        "model": "embedding-model",
        "input": "hello world",
    }

    result = await router.route_embeddings(request)

    assert result == [0.9, 0.8, 0.7]

    provider_a.embeddings.assert_awaited_once()
    provider_b.embeddings.assert_awaited_once()

    routing_resolver.resolve.assert_called_once_with(
        capability="embeddings",
        model="embedding-model",
        requested_provider=None,
    )


@pytest.mark.asyncio
async def test_embeddings_first_provider_succeeds_second_never_called():
    provider_a = FakeProvider(
        "provider-a",
        embeddings_result=[
            0.1,
            0.2,
            0.3,
        ],
    )

    provider_b = FakeProvider(
        "provider-b",
        embeddings_result=[
            0.4,
            0.5,
            0.6,
        ],
    )

    router, routing_resolver = build_router(
        [provider_a, provider_b],
    )

    request = {
        "model": "embedding-model",
        "input": "hello",
    }

    result = await router.route_embeddings(request)

    assert result == [
        0.1,
        0.2,
        0.3,
    ]

    provider_a.embeddings.assert_awaited_once()
    provider_b.embeddings.assert_not_awaited()


@pytest.mark.asyncio
async def test_stream_does_not_use_fallback_executor(
    capability_service,
):
    provider_a = FakeProvider(
        "provider-a",
        stream_chunks=[
            "chunk-a-1",
            "chunk-a-2",
        ],
    )

    provider_b = FakeProvider(
        "provider-b",
        stream_chunks=[
            "chunk-b-1",
            "chunk-b-2",
        ],
    )

    fallback_executor = RecordingFallbackExecutor()
    fallback_executor.execute = AsyncMock(
        side_effect=AssertionError("FallbackExecutor must not be used by route_stream")
    )

    router, routing_resolver = build_router(
        [provider_a, provider_b],
        fallback_executor=fallback_executor,
    )

    request = {
        "model": "stream-model",
        "messages": [
            {
                "role": "user",
                "content": "hello",
            }
        ],
    }

    try:
        await router.route_stream(request)
    except Exception:
        pass

    fallback_executor.execute.assert_not_awaited()
