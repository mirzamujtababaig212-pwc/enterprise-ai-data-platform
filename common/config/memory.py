import os


class MemoryStoreConfig:
    """
    Centralized memory-store and retrieval configuration.
    """

    BACKEND = os.getenv("MEMORY_STORE_BACKEND", "in_memory").strip().lower()

    RERANKER = os.getenv("MEMORY_RETRIEVER_RERANKER", "none").strip().lower()
    RERANKER_MODEL_ID = os.getenv(
        "MEMORY_RETRIEVER_RERANKER_MODEL_ID",
        "jinaai/jina-reranker-v1-tiny-en",
    ).strip()
    RERANKER_ONNX_FILENAME = os.getenv(
        "MEMORY_RETRIEVER_RERANKER_ONNX_FILENAME",
        "onnx/model_int8.onnx",
    ).strip()
    RERANKER_MAX_LENGTH = int(os.getenv("MEMORY_RETRIEVER_RERANKER_MAX_LENGTH", "8192"))
    RERANKER_CANDIDATE_K = int(os.getenv("MEMORY_RETRIEVER_RERANKER_CANDIDATE_K", "20"))
