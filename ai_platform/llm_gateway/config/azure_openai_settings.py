import json
import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class AzureOpenAISettings(BaseSettings):
    """
    Configuration for the Azure OpenAI provider.

    Direct AZURE_OPENAI_* environment variables take precedence
    over PROVIDER_CREDENTIALS.
    """

    model_config = SettingsConfigDict(
        extra="ignore",
        case_sensitive=False,
    )

    api_key: str = ""
    endpoint: str = ""
    api_version: str = "2024-10-21"
    chat_deployment: str = ""
    embedding_deployment: str = ""
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


def get_azure_openai_settings() -> AzureOpenAISettings:
    """
    Load Azure OpenAI configuration.

    Direct AZURE_OPENAI_* variables take precedence.
    PROVIDER_CREDENTIALS is used as a fallback.
    """

    api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    api_version = os.getenv(
        "AZURE_OPENAI_API_VERSION",
        "2024-10-21",
    )
    chat_deployment = os.getenv(
        "AZURE_OPENAI_CHAT_DEPLOYMENT",
        "",
    )
    embedding_deployment = os.getenv(
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
        "",
    )

    credentials = _load_provider_credentials()
    azure_credentials = credentials.get("azure_openai", "")

    if isinstance(azure_credentials, dict):
        if not api_key:
            api_key = azure_credentials.get("api_key", "")

        if not endpoint:
            endpoint = azure_credentials.get("endpoint", "")

        if "AZURE_OPENAI_API_VERSION" not in os.environ:
            api_version = azure_credentials.get(
                "api_version",
                api_version,
            )

        if "AZURE_OPENAI_CHAT_DEPLOYMENT" not in os.environ:
            chat_deployment = azure_credentials.get(
                "chat_deployment",
                chat_deployment,
            )

        if "AZURE_OPENAI_EMBEDDING_DEPLOYMENT" not in os.environ:
            embedding_deployment = azure_credentials.get(
                "embedding_deployment",
                embedding_deployment,
            )

    return AzureOpenAISettings(
        api_key=api_key,
        endpoint=endpoint,
        api_version=api_version,
        chat_deployment=chat_deployment,
        embedding_deployment=embedding_deployment,
        timeout=float(
            os.getenv(
                "AZURE_OPENAI_TIMEOUT",
                "60",
            )
        ),
        max_retries=int(
            os.getenv(
                "AZURE_OPENAI_MAX_RETRIES",
                "0",
            )
        ),
    )
