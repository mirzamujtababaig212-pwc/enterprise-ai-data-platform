import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "local")
    APP_NAME: str = "Enterprise AI Platform"
    APP_VERSION: str = "1.0.0"
    LOG_LEVEL: str = "INFO"

    API_KEY: str = os.getenv("API_KEY", "")

    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv(
        "GEMINI_API_KEY",
        "",
    )
    ANTHROPIC_API_KEY: str = os.getenv(
        "ANTHROPIC_API_KEY",
        "",
    )
    AZURE_OPENAI_API_KEY: str = os.getenv(
        "AZURE_OPENAI_API_KEY",
        "",
    )

    AZURE_OPENAI_ENDPOINT: str = os.getenv(
        "AZURE_OPENAI_ENDPOINT",
        "",
    )
    KAFKA_BROKER: str = os.getenv(
        "KAFKA_BROKER",
        "kafka:29092",
    )

    POSTGRES_DB: str = os.getenv(
        "POSTGRES_DB",
        "enterprise_ai",
    )

    POSTGRES_USER: str = os.getenv(
        "POSTGRES_USER",
        "enterprise_ai",
    )

    POSTGRES_PASSWORD: str = os.getenv(
        "POSTGRES_PASSWORD",
        "enterprise_ai",
    )

    POSTGRES_HOST: str = os.getenv(
        "POSTGRES_HOST",
        "postgres",
    )

    POSTGRES_PORT: int = int(
        os.getenv(
            "POSTGRES_PORT",
            "5432",
        )
    )

    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_DEFAULT_REGION: str = "us-east-1"

    OLLAMA_BASE_URL: str = "http://localhost:11434"

    DEFAULT_PROVIDER: str = "openai"
    DEFAULT_CHAT_MODEL: str = "openai-gpt"
    DEFAULT_EMBEDDING_MODEL: str = "openai-embedding"

    REQUEST_TIMEOUT: int = 60

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()
