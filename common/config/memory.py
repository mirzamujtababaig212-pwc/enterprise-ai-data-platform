import os


class MemoryStoreConfig:
    """
    Centralized memory-store backend configuration.
    """

    BACKEND = os.getenv("MEMORY_STORE_BACKEND", "in_memory").strip().lower()
