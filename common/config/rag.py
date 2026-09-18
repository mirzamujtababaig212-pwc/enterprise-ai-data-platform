import os


class RAGRetrievalConfig:
    """
    Centralized RAG retrieval and reranking configuration.
    """

    RERANKER = os.getenv("RAG_RETRIEVER_RERANKER", "none").strip().lower()
    RERANKER_MODEL_ID = os.getenv(
        "RAG_RETRIEVER_RERANKER_MODEL_ID",
        "jinaai/jina-reranker-v1-tiny-en",
    ).strip()
    RERANKER_ONNX_FILENAME = os.getenv(
        "RAG_RETRIEVER_RERANKER_ONNX_FILENAME",
        "onnx/model_int8.onnx",
    ).strip()
    RERANKER_MAX_LENGTH = int(os.getenv("RAG_RETRIEVER_RERANKER_MAX_LENGTH", "8192"))
    RERANKER_CANDIDATE_K = int(os.getenv("RAG_RETRIEVER_RERANKER_CANDIDATE_K", "20"))
