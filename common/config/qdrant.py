import os


class QdrantConfig:
    """
    Centralized Qdrant vector-store configuration.

    Environment variables:
        QDRANT_URL
        QDRANT_COLLECTION
        QDRANT_API_KEY
        QDRANT_TIMEOUT
    """

    URL = os.getenv("QDRANT_URL", "http://localhost:6333")

    COLLECTION = os.getenv(
        "QDRANT_COLLECTION",
        "enterprise_ai_knowledge",
    )

    API_KEY = os.getenv("QDRANT_API_KEY")

    TIMEOUT = float(os.getenv("QDRANT_TIMEOUT", "10.0"))
