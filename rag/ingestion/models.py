from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RAGIngestionResult:
    """
    Summary of a RAG ingestion operation.

    The result intentionally captures only the core ingestion metrics.
    Source lineage, versioning, timing, failures, and persistence can
    be added later without changing the ingestion service contract.
    """

    documents_processed: int
    documents_indexed: int
    chunks_created: int
