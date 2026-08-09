from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "Enterprise AI Platform"
    APP_VERSION: str = "1.0.0"
    LOG_LEVEL: str = "INFO"

    API_KEY: str

    OPENAI_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""

    AZURE_OPENAI_API_KEY: str = ""
    AZURE_OPENAI_ENDPOINT: str = ""

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
