import os


class VectorStoreConfig:
    """
    Centralized vector-store backend configuration.
    """

    BACKEND = os.getenv("VECTOR_STORE_BACKEND", "in_memory").strip().lower()
