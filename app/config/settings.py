from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Settings:
    environment: str
    aws_region: str
    default_provider: str
    log_level: str
    provider_credentials: dict[str, Any]

    @classmethod
    def from_environment(cls) -> "Settings":
        credentials_raw = os.getenv("PROVIDER_CREDENTIALS", "{}")

        try:
            credentials = json.loads(credentials_raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("PROVIDER_CREDENTIALS must contain valid JSON") from exc

        return cls(
            environment=os.getenv("ENVIRONMENT", "dev"),
            aws_region=os.getenv("AWS_REGION", "us-east-1"),
            default_provider=os.getenv(
                "DEFAULT_PROVIDER",
                "openai",
            ).lower(),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            provider_credentials=credentials,
        )
