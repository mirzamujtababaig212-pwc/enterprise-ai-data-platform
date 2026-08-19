import json
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

if not OPENAI_API_KEY:
    raw_credentials = os.getenv("PROVIDER_CREDENTIALS", "")

    if raw_credentials:
        try:
            credentials = json.loads(raw_credentials)
            provider_key = credentials.get("openai", "")

            if isinstance(provider_key, str):
                OPENAI_API_KEY = provider_key
        except (json.JSONDecodeError, TypeError):
            OPENAI_API_KEY = ""

OPENAI_BASE_URL = os.getenv(
    "OPENAI_BASE_URL",
    "https://api.openai.com/v1",
)

OPENAI_MODEL = os.getenv(
    "OPENAI_MODEL",
    "gpt-4.1-mini",
)


def validate_openai_settings():
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is missing.")
