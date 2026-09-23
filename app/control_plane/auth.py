import hashlib

from fastapi import Request
from fastapi.responses import JSONResponse
from opentelemetry import trace

from ai_platform.llm_gateway.config.settings import settings
from app.control_plane.identity import AuthenticatedIdentity

tracer = trace.get_tracer(__name__)

API_KEY_NAME = "x-api-key"

PUBLIC_PATHS = {
    "/api/v1/health",
    "/api/v1/platform/health",
    "/metrics",
    "/docs",
    "/redoc",
    "/openapi.json",
}


def get_valid_api_keys() -> set[str]:
    return {key.strip() for key in settings.API_KEY.split(",") if key.strip()}


def principal_from_api_key(api_key: str) -> str:
    digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
    return f"api_key:{digest}"


def tenant_id_from_api_key(api_key: str) -> str:
    """Resolve an API key to its explicitly configured tenant.

    Tenant identity is never accepted from request metadata.
    """
    configured = getattr(settings, "CONTROL_PLANE_API_KEY_TENANTS", "").strip()

    if not configured:
        raise ValueError(
            "CONTROL_PLANE_API_KEY_TENANTS must be configured " "for authenticated tenant identity."
        )

    for entry in configured.split(","):
        entry = entry.strip()
        if not entry:
            continue

        tenant_id, separator, configured_key = entry.partition("=")

        if not separator:
            continue

        if configured_key.strip() == api_key:
            tenant_id = tenant_id.strip()
            if tenant_id:
                return tenant_id

    raise ValueError("Authenticated API key has no configured tenant.")


def identity_from_api_key(api_key: str) -> AuthenticatedIdentity:
    return AuthenticatedIdentity(
        principal=principal_from_api_key(api_key),
        tenant_id=tenant_id_from_api_key(api_key),
    )


class ControlPlaneAPIKeyMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)
        path = request.url.path

        if path in PUBLIC_PATHS:
            await self.app(scope, receive, send)
            return

        api_key = request.headers.get(API_KEY_NAME)
        valid_api_keys = get_valid_api_keys()

        if api_key not in valid_api_keys:
            response = JSONResponse(
                status_code=401,
                content={
                    "detail": "Invalid or missing API key",
                },
                headers={
                    "WWW-Authenticate": "ApiKey",
                },
            )
            await response(scope, receive, send)
            return

        try:
            identity = identity_from_api_key(api_key)
        except ValueError:
            response = JSONResponse(
                status_code=403,
                content={
                    "detail": "Authenticated API key is not mapped to a tenant",
                },
            )
            await response(scope, receive, send)
            return

        state = scope.setdefault("state", {})
        state["principal"] = identity.principal
        state["tenant_id"] = identity.tenant_id
        state["identity"] = identity

        with tracer.start_as_current_span("control_plane.authentication"):
            await self.app(scope, receive, send)
