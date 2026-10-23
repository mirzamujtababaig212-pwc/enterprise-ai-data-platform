from contextlib import contextmanager
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from ai_platform.agents.policy import TenantPolicy, TenantPolicyEngine
from app.control_plane.app import app
from app.control_plane.dependencies import get_usage_store
from app.control_plane.routes.llm import _authorized_provider_allow_list

client = TestClient(app)

AUTH_HEADERS = {"x-api-key": "super-secret-key"}


@contextmanager
def _isolated_usage_store():
    """Prevent direct-LLM authorization tests from requiring PostgreSQL."""
    previous = app.dependency_overrides.get(get_usage_store)
    fake_store = Mock()
    app.dependency_overrides[get_usage_store] = lambda: fake_store
    try:
        yield fake_store
    finally:
        if previous is None:
            app.dependency_overrides.pop(get_usage_store, None)
        else:
            app.dependency_overrides[get_usage_store] = previous


def test_chat_route_rejects_missing_required_fields() -> None:
    response = client.post(
        "/api/v1/llm/chat",
        json={},
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 422


def test_embeddings_route_rejects_missing_required_fields() -> None:
    response = client.post(
        "/api/v1/llm/embeddings",
        json={},
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 422


def test_chat_records_usage_event(usage_store) -> None:
    fake_router = AsyncMock()
    fake_router.route_chat_with_metadata.return_value = type(
        "ChatResult",
        (),
        {
            "response": {
                "reply": "Hello from the model",
                "usage": {
                    "tokens_in": 10,
                    "tokens_out": 20,
                },
            },
            "provider_name": "openai",
            "model_name": "gpt-4.1-mini",
        },
    )()

    with patch(
        "app.control_plane.routes.llm.get_llm_router",
        return_value=fake_router,
    ):
        response = client.post(
            "/api/v1/llm/chat",
            json={
                "prompt": "Hello",
                "provider": "openai",
                "model": "gpt-4.1-mini",
            },
            headers=AUTH_HEADERS,
        )

    assert response.status_code == 200

    body = response.json()

    assert body["reply"] == "Hello from the model"

    request_id = body["metrics"]["request_id"]

    events = usage_store.list(
        request_id=request_id,
    )

    assert len(events) == 1

    event = events[0]

    assert event.request_id == request_id
    assert event.capability == "llm.chat"
    assert event.provider == "openai"
    assert event.model == "gpt-4.1-mini"
    assert event.tokens_in == 10
    assert event.tokens_out == 20
    assert event.status == "success"
    assert event.latency_ms >= 0


def test_chat_records_prometheus_usage_metrics(usage_store) -> None:
    fake_router = AsyncMock()
    fake_router.route_chat_with_metadata.return_value = type(
        "ChatResult",
        (),
        {
            "response": {
                "reply": "Telemetry response",
                "usage": {
                    "tokens_in": 10,
                    "tokens_out": 20,
                },
            },
            "provider_name": "openai",
            "model_name": "gpt-4.1-mini",
        },
    )()

    with (
        patch(
            "app.control_plane.routes.llm.get_llm_router",
            return_value=fake_router,
        ),
        patch(
            "app.control_plane.routes.llm.MODEL_REQUESTS_TOTAL",
        ) as model_requests,
        patch(
            "app.control_plane.routes.llm.INPUT_TOKENS_TOTAL",
        ) as input_tokens,
        patch(
            "app.control_plane.routes.llm.OUTPUT_TOKENS_TOTAL",
        ) as output_tokens,
        patch(
            "app.control_plane.routes.llm.ESTIMATED_COST_TOTAL",
        ) as estimated_cost,
    ):
        response = client.post(
            "/api/v1/llm/chat",
            json={
                "prompt": "Hello",
                "provider": "openai",
                "model": "gpt-4.1-mini",
            },
            headers=AUTH_HEADERS,
        )

    assert response.status_code == 200

    model_requests.labels.assert_called_once_with(
        model="gpt-4.1-mini",
    )
    model_requests.labels.return_value.inc.assert_called_once_with()

    input_tokens.inc.assert_called_once_with(10)
    output_tokens.inc.assert_called_once_with(20)
    expected_cost = Decimal("10") / Decimal("1000000") * Decimal("0.40") + Decimal("20") / Decimal(
        "1000000"
    ) * Decimal("1.60")

    estimated_cost.inc.assert_called_once_with(
        float(expected_cost),
    )


def test_chat_records_actual_fallback_provider(usage_store) -> None:
    fake_router = AsyncMock()
    fake_router.route_chat_with_metadata.return_value = type(
        "ChatResult",
        (),
        {
            "response": {
                "reply": "Fallback response",
                "usage": {
                    "tokens_in": 5,
                    "tokens_out": 8,
                },
            },
            "provider_name": "fallback-provider",
            "model_name": "gpt-4.1-mini",
        },
    )()

    with patch(
        "app.control_plane.routes.llm.get_llm_router",
        return_value=fake_router,
    ):
        response = client.post(
            "/api/v1/llm/chat",
            json={
                "prompt": "Hello",
                "provider": "openai",
                "model": "gpt-4.1-mini",
            },
            headers=AUTH_HEADERS,
        )

    assert response.status_code == 200

    request_id = response.json()["metrics"]["request_id"]

    events = usage_store.list(
        request_id=request_id,
    )

    assert len(events) == 1
    assert events[0].provider == "fallback-provider"


def test_chat_records_error_usage_event(usage_store) -> None:
    fake_router = AsyncMock()
    fake_router.route_chat_with_metadata.side_effect = ValueError("Invalid chat request")

    with patch(
        "app.control_plane.routes.llm.get_llm_router",
        return_value=fake_router,
    ):
        response = client.post(
            "/api/v1/llm/chat",
            json={
                "prompt": "Hello",
                "provider": "openai",
                "model": "gpt-4.1-mini",
            },
            headers=AUTH_HEADERS,
        )

    assert response.status_code == 400

    events = usage_store.list()

    assert len(events) == 1

    event = events[0]

    assert event.capability == "llm.chat"
    assert event.provider == "openai"
    assert event.model == "gpt-4.1-mini"
    assert event.status == "error"
    assert event.tokens_in == 0
    assert event.tokens_out == 0


def test_embeddings_records_usage_event(usage_store) -> None:
    fake_router = AsyncMock()
    fake_router.route_embeddings_with_metadata.return_value = type(
        "EmbeddingResult",
        (),
        {
            "response": [0.1, 0.2, 0.3],
            "provider_name": "openai",
        },
    )()

    with patch(
        "app.control_plane.routes.llm.get_llm_router",
        return_value=fake_router,
    ):
        response = client.post(
            "/api/v1/llm/embeddings",
            json={
                "text": "Hello world",
                "provider": "openai",
                "model": "gpt-4.1-mini",
            },
            headers=AUTH_HEADERS,
        )

    assert response.status_code == 200

    body = response.json()

    assert body["vector"] == [0.1, 0.2, 0.3]

    request_id = body["metrics"]["request_id"]

    events = usage_store.list(
        request_id=request_id,
    )

    assert len(events) == 1

    event = events[0]

    assert event.request_id == request_id
    assert event.capability == "llm.embeddings"
    assert event.provider == "openai"
    assert event.model == "gpt-4.1-mini"
    assert event.tokens_in == 2
    assert event.tokens_out == 0
    assert event.status == "success"
    assert event.latency_ms >= 0


def test_embeddings_records_prometheus_usage_metrics(usage_store) -> None:
    fake_router = AsyncMock()
    fake_router.route_embeddings_with_metadata.return_value = type(
        "EmbeddingResult",
        (),
        {
            "response": [0.1, 0.2, 0.3],
            "provider_name": "openai",
        },
    )()

    with (
        patch(
            "app.control_plane.routes.llm.get_llm_router",
            return_value=fake_router,
        ),
        patch(
            "app.control_plane.routes.llm.MODEL_REQUESTS_TOTAL",
        ) as model_requests,
        patch(
            "app.control_plane.routes.llm.INPUT_TOKENS_TOTAL",
        ) as input_tokens,
    ):
        response = client.post(
            "/api/v1/llm/embeddings",
            json={
                "text": "Hello world",
                "provider": "openai",
                "model": "gpt-4.1-mini",
            },
            headers=AUTH_HEADERS,
        )

    assert response.status_code == 200

    model_requests.labels.assert_called_once_with(
        model="gpt-4.1-mini",
    )
    model_requests.labels.return_value.inc.assert_called_once_with()

    input_tokens.inc.assert_called_once_with(2)


def test_embeddings_records_actual_fallback_provider(usage_store) -> None:
    fake_router = AsyncMock()
    fake_router.route_embeddings_with_metadata.return_value = type(
        "EmbeddingResult",
        (),
        {
            "response": [0.9, 0.8, 0.7],
            "provider_name": "fallback-provider",
        },
    )()

    with patch(
        "app.control_plane.routes.llm.get_llm_router",
        return_value=fake_router,
    ):
        response = client.post(
            "/api/v1/llm/embeddings",
            json={
                "text": "Hello world",
                "provider": "openai",
                "model": "gpt-4.1-mini",
            },
            headers=AUTH_HEADERS,
        )

    assert response.status_code == 200

    request_id = response.json()["metrics"]["request_id"]

    events = usage_store.list(
        request_id=request_id,
    )

    assert len(events) == 1
    assert events[0].provider == "fallback-provider"


def test_embeddings_records_error_usage_event(usage_store) -> None:
    fake_router = AsyncMock()
    fake_router.route_embeddings_with_metadata.side_effect = ValueError(
        "Embedding provider failure",
    )

    with patch(
        "app.control_plane.routes.llm.get_llm_router",
        return_value=fake_router,
    ):
        response = client.post(
            "/api/v1/llm/embeddings",
            json={
                "text": "Hello world",
                "provider": "openai",
                "model": "gpt-4.1-mini",
            },
            headers=AUTH_HEADERS,
        )

    assert response.status_code == 404

    events = usage_store.list()

    assert len(events) == 1

    event = events[0]

    assert event.capability == "llm.embeddings"
    assert event.provider == "openai"
    assert event.model == "gpt-4.1-mini"
    assert event.status == "error"


def _direct_llm_policy_engine() -> TenantPolicyEngine:
    engine = TenantPolicyEngine()
    engine.register_policy(
        TenantPolicy(
            tenant_id="tenant-a",
            allowed_providers=frozenset({"openai"}),
        )
    )
    return engine


def test_direct_llm_chat_denies_disallowed_provider() -> None:
    fake_router = AsyncMock()

    with (
        patch(
            "app.control_plane.routes.llm.get_tenant_policy_engine",
            return_value=_direct_llm_policy_engine(),
        ),
        patch(
            "app.control_plane.routes.llm.get_llm_router",
            return_value=fake_router,
        ),
    ):
        response = client.post(
            "/api/v1/llm/chat",
            json={
                "prompt": "Hello",
                "provider": "ollama",
                "model": "gpt-4.1-mini",
            },
            headers=AUTH_HEADERS,
        )

    assert response.status_code == 403
    fake_router.route_chat_with_metadata.assert_not_awaited()


def test_direct_llm_embeddings_deny_disallowed_provider() -> None:
    fake_router = AsyncMock()

    with (
        patch(
            "app.control_plane.routes.llm.get_tenant_policy_engine",
            return_value=_direct_llm_policy_engine(),
        ),
        patch(
            "app.control_plane.routes.llm.get_llm_router",
            return_value=fake_router,
        ),
    ):
        response = client.post(
            "/api/v1/llm/embeddings",
            json={
                "text": "Hello world",
                "provider": "ollama",
                "model": "gpt-4.1-mini",
            },
            headers=AUTH_HEADERS,
        )

    assert response.status_code == 403
    fake_router.route_embeddings_with_metadata.assert_not_awaited()


def test_direct_llm_chat_passes_policy_provider_allow_list() -> None:
    fake_router = AsyncMock()
    fake_router.route_chat_with_metadata.return_value = type(
        "ChatResult",
        (),
        {
            "response": {
                "reply": "Authorized response",
                "usage": {"tokens_in": 2, "tokens_out": 3},
            },
            "provider_name": "openai",
            "model_name": "gpt-4.1-mini",
        },
    )()

    with (
        _isolated_usage_store(),
        patch(
            "app.control_plane.routes.llm.get_tenant_policy_engine",
            return_value=_direct_llm_policy_engine(),
        ),
        patch(
            "app.control_plane.routes.llm.get_llm_router",
            return_value=fake_router,
        ),
    ):
        response = client.post(
            "/api/v1/llm/chat",
            json={
                "prompt": "Hello",
                "model": "gpt-4.1-mini",
                "provider": "openai",
            },
            headers=AUTH_HEADERS,
        )

    assert response.status_code == 200
    assert fake_router.route_chat_with_metadata.await_args.kwargs["allowed_providers"] == (
        "openai",
    )


def test_direct_llm_embeddings_pass_policy_provider_allow_list() -> None:
    fake_router = AsyncMock()
    fake_router.route_embeddings_with_metadata.return_value = type(
        "EmbeddingResult",
        (),
        {
            "response": [0.1, 0.2, 0.3],
            "provider_name": "openai",
        },
    )()

    with (
        _isolated_usage_store(),
        patch(
            "app.control_plane.routes.llm.get_tenant_policy_engine",
            return_value=_direct_llm_policy_engine(),
        ),
        patch(
            "app.control_plane.routes.llm.get_llm_router",
            return_value=fake_router,
        ),
    ):
        response = client.post(
            "/api/v1/llm/embeddings",
            json={
                "text": "Hello world",
                "model": "gpt-4.1-mini",
                "provider": "openai",
            },
            headers=AUTH_HEADERS,
        )

    assert response.status_code == 200
    assert fake_router.route_embeddings_with_metadata.await_args.kwargs["allowed_providers"] == (
        "openai",
    )


def test_direct_llm_chat_denies_model_outside_policy() -> None:
    engine = TenantPolicyEngine()
    engine.register_policy(
        TenantPolicy(
            tenant_id="tenant-a",
            allowed_models=frozenset({"gpt-5"}),
            allowed_providers=frozenset({"openai"}),
        )
    )
    fake_router = AsyncMock()

    with (
        patch(
            "app.control_plane.routes.llm.get_tenant_policy_engine",
            return_value=engine,
        ),
        patch(
            "app.control_plane.routes.llm.get_llm_router",
            return_value=fake_router,
        ),
    ):
        response = client.post(
            "/api/v1/llm/chat",
            json={
                "prompt": "Hello",
                "model": "gpt-4.1-mini",
                "provider": "openai",
            },
            headers=AUTH_HEADERS,
        )

    assert response.status_code == 403
    fake_router.route_chat_with_metadata.assert_not_awaited()


def test_direct_llm_embeddings_denies_model_outside_policy() -> None:
    engine = TenantPolicyEngine()
    engine.register_policy(
        TenantPolicy(
            tenant_id="tenant-a",
            allowed_models=frozenset({"gpt-5"}),
            allowed_providers=frozenset({"openai"}),
        )
    )
    fake_router = AsyncMock()

    with (
        patch(
            "app.control_plane.routes.llm.get_tenant_policy_engine",
            return_value=engine,
        ),
        patch(
            "app.control_plane.routes.llm.get_llm_router",
            return_value=fake_router,
        ),
    ):
        response = client.post(
            "/api/v1/llm/embeddings",
            json={
                "text": "Hello world",
                "model": "gpt-4.1-mini",
                "provider": "openai",
            },
            headers=AUTH_HEADERS,
        )

    assert response.status_code == 403
    fake_router.route_embeddings_with_metadata.assert_not_awaited()


@pytest.mark.parametrize(
    ("path", "payload", "router_method"),
    [
        (
            "/api/v1/llm/chat",
            {"prompt": "Hello", "model": "gpt-5", "provider": "openai"},
            "route_chat_with_metadata",
        ),
        (
            "/api/v1/llm/embeddings",
            {"text": "Hello world", "model": "gpt-5", "provider": "openai"},
            "route_embeddings_with_metadata",
        ),
    ],
)
def test_direct_llm_endpoints_deny_missing_trusted_tenant_identity(
    path: str,
    payload: dict[str, str],
    router_method: str,
) -> None:
    fake_router = AsyncMock()
    identity_without_tenant = SimpleNamespace(
        principal="api_key:test",
        tenant_id=" ",
    )

    with (
        patch(
            "app.control_plane.auth.identity_from_api_key",
            return_value=identity_without_tenant,
        ),
        patch(
            "app.control_plane.routes.llm.get_tenant_policy_engine",
            return_value=_direct_llm_policy_engine(),
        ),
        patch(
            "app.control_plane.routes.llm.get_llm_router",
            return_value=fake_router,
        ),
    ):
        response = client.post(path, json=payload, headers=AUTH_HEADERS)

    assert response.status_code == 403
    assert response.json()["detail"] == ("Authenticated tenant identity is required.")
    getattr(fake_router, router_method).assert_not_awaited()


@pytest.mark.parametrize(
    ("path", "payload", "router_method"),
    [
        (
            "/api/v1/llm/chat",
            {"prompt": "Hello", "model": "gpt-5", "provider": "openai"},
            "route_chat_with_metadata",
        ),
        (
            "/api/v1/llm/embeddings",
            {"text": "Hello world", "model": "gpt-5", "provider": "openai"},
            "route_embeddings_with_metadata",
        ),
    ],
)
def test_direct_llm_endpoints_deny_tenant_without_registered_policy(
    path: str,
    payload: dict[str, str],
    router_method: str,
) -> None:
    fake_router = AsyncMock()
    identity_without_policy = SimpleNamespace(
        principal="api_key:test",
        tenant_id="tenant-unregistered",
    )

    with (
        patch(
            "app.control_plane.auth.identity_from_api_key",
            return_value=identity_without_policy,
        ),
        patch(
            "app.control_plane.routes.llm.get_tenant_policy_engine",
            return_value=_direct_llm_policy_engine(),
        ),
        patch(
            "app.control_plane.routes.llm.get_llm_router",
            return_value=fake_router,
        ),
    ):
        response = client.post(path, json=payload, headers=AUTH_HEADERS)

    assert response.status_code == 403
    getattr(fake_router, router_method).assert_not_awaited()


def test_direct_llm_authorization_requires_tenant_identity() -> None:
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/v1/llm/chat",
            "headers": [],
            "state": {},
        }
    )

    with (
        patch(
            "app.control_plane.routes.llm.get_tenant_policy_engine",
            return_value=_direct_llm_policy_engine(),
        ),
        pytest.raises(HTTPException) as exc_info,
    ):
        _authorized_provider_allow_list(
            request,
            model="gpt-5",
            provider="openai",
        )

    assert exc_info.value.status_code == 403
    assert exc_info.value.detail == "Authenticated tenant identity is required."


def test_direct_llm_authorization_fails_closed_when_required_engine_missing() -> None:
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/v1/llm/chat",
            "headers": [],
            "state": {"tenant_id": "tenant-a"},
        }
    )

    with (
        patch(
            "app.control_plane.routes.llm.get_tenant_policy_engine",
            return_value=None,
        ),
        patch(
            "app.control_plane.routes.llm.tenant_policy_enforcement_required",
            return_value=True,
        ),
        pytest.raises(HTTPException) as exc_info,
    ):
        _authorized_provider_allow_list(
            request,
            model="gpt-5",
            provider="openai",
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Tenant policy engine is unavailable."


def test_direct_llm_authorization_allows_disabled_policy_in_permitted_environment() -> None:
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/v1/llm/chat",
            "headers": [],
            "state": {},
        }
    )

    with (
        patch(
            "app.control_plane.routes.llm.get_tenant_policy_engine",
            return_value=None,
        ),
        patch(
            "app.control_plane.routes.llm.tenant_policy_enforcement_required",
            return_value=False,
        ),
    ):
        result = _authorized_provider_allow_list(
            request,
            model="gpt-5",
            provider="openai",
        )

    assert result is None


async def _fake_stream_chunks():
    yield "hello"
    yield " world"


def test_direct_llm_stream_passes_policy_allow_list() -> None:
    fake_router = Mock()
    fake_router.route_stream = Mock(return_value=_fake_stream_chunks())

    with (
        patch(
            "app.control_plane.routes.llm.get_tenant_policy_engine",
            return_value=_direct_llm_policy_engine(),
        ),
        patch(
            "app.control_plane.routes.llm.get_llm_router",
            return_value=fake_router,
        ),
    ):
        response = client.post(
            "/api/v1/llm/chat/stream",
            json={
                "prompt": "Hello",
                "provider": "openai",
                "model": "gpt-4.1-mini",
                "stream": True,
            },
            headers=AUTH_HEADERS,
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "data: hello\n\n" in response.text
    assert "data:  world\n\n" in response.text
    assert "data: [DONE]\n\n" in response.text

    fake_router.route_stream.assert_called_once()
    assert fake_router.route_stream.call_args.kwargs["allowed_providers"] == ("openai",)


async def _fake_multiline_stream_chunks():
    yield "first line\nsecond line"


async def _fake_failing_stream_chunks():
    yield "partial output"
    raise RuntimeError("sensitive-provider-error")


def test_direct_llm_stream_frames_multiline_chunks() -> None:
    fake_router = Mock()
    fake_router.route_stream = Mock(return_value=_fake_multiline_stream_chunks())

    with (
        patch(
            "app.control_plane.routes.llm.get_tenant_policy_engine",
            return_value=_direct_llm_policy_engine(),
        ),
        patch(
            "app.control_plane.routes.llm.get_llm_router",
            return_value=fake_router,
        ),
    ):
        response = client.post(
            "/api/v1/llm/chat/stream",
            json={
                "prompt": "Hello",
                "provider": "openai",
                "model": "gpt-4.1-mini",
                "stream": True,
            },
            headers=AUTH_HEADERS,
        )

    assert response.status_code == 200
    assert "data: first line\ndata: second line\n\n" in response.text
    assert "data: [DONE]\n\n" in response.text


def test_direct_llm_stream_sanitizes_generation_failure() -> None:
    fake_router = Mock()
    fake_router.route_stream = Mock(return_value=_fake_failing_stream_chunks())

    with (
        patch(
            "app.control_plane.routes.llm.get_tenant_policy_engine",
            return_value=_direct_llm_policy_engine(),
        ),
        patch(
            "app.control_plane.routes.llm.get_llm_router",
            return_value=fake_router,
        ),
    ):
        response = client.post(
            "/api/v1/llm/chat/stream",
            json={
                "prompt": "Hello",
                "provider": "openai",
                "model": "gpt-4.1-mini",
                "stream": True,
            },
            headers=AUTH_HEADERS,
        )

    assert response.status_code == 200
    assert "data: partial output\n\n" in response.text
    assert 'event: error\ndata: {"error":"Stream generation failed."}\n\n' in response.text
    assert "sensitive-provider-error" not in response.text
    assert "data: [DONE]\n\n" not in response.text


def test_direct_llm_stream_denies_disallowed_provider() -> None:
    fake_router = Mock()
    fake_router.route_stream = Mock(return_value=_fake_stream_chunks())

    with (
        patch(
            "app.control_plane.routes.llm.get_tenant_policy_engine",
            return_value=_direct_llm_policy_engine(),
        ),
        patch(
            "app.control_plane.routes.llm.get_llm_router",
            return_value=fake_router,
        ),
    ):
        response = client.post(
            "/api/v1/llm/chat/stream",
            json={
                "prompt": "Hello",
                "provider": "ollama",
                "model": "gpt-4.1-mini",
                "stream": True,
            },
            headers=AUTH_HEADERS,
        )

    assert response.status_code == 403
    fake_router.route_stream.assert_not_called()


def test_direct_llm_stream_fails_closed_when_required_engine_missing() -> None:
    fake_router = Mock()
    fake_router.route_stream = Mock(return_value=_fake_stream_chunks())

    with (
        patch(
            "app.control_plane.routes.llm.get_tenant_policy_engine",
            return_value=None,
        ),
        patch(
            "app.control_plane.routes.llm.tenant_policy_enforcement_required",
            return_value=True,
        ),
        patch(
            "app.control_plane.routes.llm.get_llm_router",
            return_value=fake_router,
        ),
    ):
        response = client.post(
            "/api/v1/llm/chat/stream",
            json={
                "prompt": "Hello",
                "provider": "openai",
                "model": "gpt-4.1-mini",
                "stream": True,
            },
            headers=AUTH_HEADERS,
        )

    assert response.status_code == 503
    assert response.json()["detail"] == "Tenant policy engine is unavailable."
    fake_router.route_stream.assert_not_called()


def test_direct_llm_stream_requires_stream_true() -> None:
    fake_router = Mock()
    fake_router.route_stream = Mock(return_value=_fake_stream_chunks())

    with patch(
        "app.control_plane.routes.llm.get_llm_router",
        return_value=fake_router,
    ):
        response = client.post(
            "/api/v1/llm/chat/stream",
            json={
                "prompt": "Hello",
                "provider": "openai",
                "model": "gpt-4.1-mini",
                "stream": False,
            },
            headers=AUTH_HEADERS,
        )

    assert response.status_code == 400
    assert response.json()["detail"] == ("Set stream=true to use the streaming endpoint.")
    fake_router.route_stream.assert_not_called()
