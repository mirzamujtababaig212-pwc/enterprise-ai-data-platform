import os
from dataclasses import dataclass


@dataclass(frozen=True)
class GeminiSettings:
    api_key: str = ""
    model: str = "gemini-2.5-flash"
    embedding_model: str = "gemini-embedding-001"
    timeout: float = 60.0


def get_gemini_settings() -> GeminiSettings:
    return GeminiSettings(
        api_key=os.getenv("GEMINI_API_KEY", ""),
        model=os.getenv(
            "GEMINI_MODEL",
            "gemini-2.5-flash",
        ),
        embedding_model=os.getenv(
            "GEMINI_EMBEDDING_MODEL",
            "gemini-embedding-001",
        ),
        timeout=float(
            os.getenv(
                "GEMINI_TIMEOUT",
                "60.0",
            )
        ),
    )
