import json
import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class OpenAISettings(BaseSettings):
    """
    Configuration for the OpenAI provider.

    Direct environment variables take precedence over
    PROVIDER_CREDENTIALS.
    """

    model_config = SettingsConfigDict(
        extra="ignore",
        case_sensitive=False,
    )

    api_key: str = ""
    model: str = "gpt-4.1-mini"
    base_url: str = "https://api.openai.com/v1"
    timeout: float = 60.0
    max_retries: int = 0


def _load_provider_credentials() -> dict:
    raw_credentials = os.getenv("PROVIDER_CREDENTIALS", "")

    if not raw_credentials:
        return {}

    try:
        credentials = json.loads(raw_credentials)
    except (json.JSONDecodeError, TypeError):
        return {}

    if not isinstance(credentials, dict):
        return {}

    return credentials


def get_openai_settings() -> OpenAISettings:
    """
    Load OpenAI configuration.

    Direct OPENAI_* environment variables take precedence.
    PROVIDER_CREDENTIALS is used as a fallback.
    """

    api_key = os.getenv("OPENAI_API_KEY", "")
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    base_url = os.getenv(
        "OPENAI_BASE_URL",
        "https://api.openai.com/v1",
    )

    if not api_key:
        credentials = _load_provider_credentials()
        openai_credentials = credentials.get("openai", "")

        if isinstance(openai_credentials, dict):
            api_key = openai_credentials.get(
                "api_key",
                "",
            )

            if "OPENAI_MODEL" not in os.environ:
                model = openai_credentials.get(
                    "model",
                    model,
                )

            if "OPENAI_BASE_URL" not in os.environ:
                base_url = openai_credentials.get(
                    "base_url",
                    base_url,
                )

        elif isinstance(openai_credentials, str):
            api_key = openai_credentials

    return OpenAISettings(
        api_key=api_key,
        model=model,
        base_url=base_url,
        timeout=float(
            os.getenv(
                "OPENAI_TIMEOUT",
                "60",
            )
        ),
        max_retries=int(
            os.getenv(
                "OPENAI_MAX_RETRIES",
                "0",
            )
        ),
    )


def validate_openai_settings() -> None:
    settings = get_openai_settings()

    if not settings.api_key:
        raise ValueError(
            "OPENAI_API_KEY is missing and no OpenAI credentials "
            "were found in PROVIDER_CREDENTIALS."
        )
