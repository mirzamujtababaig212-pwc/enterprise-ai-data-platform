from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(
    name="X-API-Key",
    scheme_name="ApiKey",
    description="API key required to access protected LLM Gateway endpoints.",
    auto_error=False,
)
