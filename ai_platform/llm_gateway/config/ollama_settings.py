import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class OllamaSettings(BaseSettings):
    """
    Configuration for the Ollama provider.

    Direct OLLAMA_* environment variables are used for provider
    configuration. No API key is required for the local Ollama API.
    """

    model_config = SettingsConfigDict(
        extra="ignore",
        case_sensitive=False,
    )

    base_url: str = "http://localhost:11434"
    chat_model: str = "llama3.2"
    embedding_model: str = "nomic-embed-text"
    timeout: float = 60.0


def get_ollama_settings() -> OllamaSettings:
    """Load Ollama configuration from environment variables."""

    return OllamaSettings(
        base_url=os.getenv(
            "OLLAMA_BASE_URL",
            "http://localhost:11434",
        ).rstrip("/"),
        chat_model=os.getenv(
            "OLLAMA_CHAT_MODEL",
            "llama3.2",
        ),
        embedding_model=os.getenv(
            "OLLAMA_EMBEDDING_MODEL",
            "nomic-embed-text",
        ),
        timeout=float(
            os.getenv(
                "OLLAMA_TIMEOUT",
                "60",
            )
        ),
    )
