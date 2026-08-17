from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RuntimeSettings(BaseSettings):
    """
    Production runtime configuration.
    """

    environment: str = Field(
        default="development",
    )

    service_name: str = Field(
        default="enterprise-ai-llm-gateway",
    )

    service_version: str = Field(
        default="0.1.0",
    )

    log_level: str = Field(
        default="INFO",
    )

    request_timeout_seconds: float = Field(
        default=60.0,
        gt=0,
    )

    provider_timeout_seconds: float = Field(
        default=45.0,
        gt=0,
    )

    max_retries: int = Field(
        default=2,
        ge=0,
        le=10,
    )

    enable_tracing: bool = True

    enable_metrics: bool = True

    enable_request_logging: bool = True

    model_registry_refresh_seconds: int = Field(
        default=300,
        ge=0,
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_runtime_settings() -> RuntimeSettings:
    return RuntimeSettings()
