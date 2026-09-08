import os
from dataclasses import dataclass


@dataclass(frozen=True)
class BedrockSettings:
    access_key_id: str = ""
    secret_access_key: str = ""
    region: str = "us-east-1"
    chat_model: str = "amazon.nova-micro-v1:0"
    embedding_model: str = "amazon.titan-embed-text-v2:0"
    embedding_dimensions: int = 1024
    embedding_normalize: bool = True
    timeout: float = 60.0


def get_bedrock_settings() -> BedrockSettings:
    return BedrockSettings(
        access_key_id=os.getenv("DELDAI_AWS_ACCESS_KEY_ID", ""),
        secret_access_key=os.getenv("DELDAI_AWS_SECRET_ACCESS_KEY", ""),
        region=os.getenv("DELDAI_AWS_DEFAULT_REGION", "us-east-1"),
        chat_model=os.getenv(
            "DELDAI_BEDROCK_CHAT_MODEL",
            "amazon.nova-micro-v1:0",
        ),
        embedding_model=os.getenv(
            "DELDAI_BEDROCK_EMBEDDING_MODEL",
            "amazon.titan-embed-text-v2:0",
        ),
        embedding_dimensions=int(os.getenv("DELDAI_BEDROCK_EMBEDDING_DIMENSIONS", "1024")),
        embedding_normalize=os.getenv(
            "DELDAI_BEDROCK_EMBEDDING_NORMALIZE",
            "true",
        )
        .strip()
        .lower()
        in {"1", "true", "yes", "on"},
        timeout=float(os.getenv("DELDAI_BEDROCK_TIMEOUT", "60.0")),
    )
