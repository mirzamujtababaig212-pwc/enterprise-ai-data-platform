import json
import os


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

if not OPENAI_API_KEY:
    raw_credentials = os.getenv("PROVIDER_CREDENTIALS", "")

    if raw_credentials:
        try:
            credentials = json.loads(raw_credentials)
            openai_val = credentials.get("openai", "")

            if isinstance(openai_val, dict):
                OPENAI_API_KEY = openai_val.get("api_key", "")

                if "OPENAI_MODEL" not in os.environ:
                    OPENAI_MODEL = openai_val.get(
                        "model",
                        "gpt-4.1-mini",
                    )

            elif isinstance(openai_val, str):
                OPENAI_API_KEY = openai_val

        except (json.JSONDecodeError, TypeError):
            OPENAI_API_KEY = ""


OPENAI_BASE_URL = os.getenv(
    "OPENAI_BASE_URL",
    "https://api.openai.com/v1",
)


def validate_openai_settings():
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is missing.")
