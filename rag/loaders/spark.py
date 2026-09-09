from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from pyspark.sql import DataFrame

from rag.models import Document

RowFormatter = Callable[[dict[str, Any]], str]


class SparkDataFrameDocumentLoader:
    """
    Convert a Spark DataFrame into canonical RAG Documents.

    The loader is intentionally generic:
    - one Spark row becomes one Document
    - document IDs are supplied by the caller
    - row content is produced by a caller-provided formatter
    - metadata is produced by a caller-provided function

    This adapter bridges tabular enterprise data into the existing
    RAGIndexer without changing the RAG core contracts.
    """

    def __init__(
        self,
        *,
        id_fn: Callable[[dict[str, Any]], str],
        content_fn: RowFormatter,
        metadata_fn: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> None:
        self._id_fn = id_fn
        self._content_fn = content_fn
        self._metadata_fn = metadata_fn

    def load(self, dataframe: DataFrame) -> Sequence[Document]:
        if dataframe is None:
            raise ValueError("dataframe must not be None")

        documents: list[Document] = []

        for row in dataframe.toLocalIterator():
            values = row.asDict(recursive=True)

            document_id = self._id_fn(values)

            if not document_id or not document_id.strip():
                raise ValueError("id_fn must return a non-empty document ID")

            content = self._content_fn(values)

            if not isinstance(content, str) or not content.strip():
                raise ValueError(
                    f"content_fn must return non-empty text for document " f"{document_id!r}"
                )

            metadata = self._metadata_fn(values)

            if metadata is None:
                metadata = {}

            documents.append(
                Document(
                    id=document_id,
                    content=content,
                    metadata=dict(metadata),
                )
            )

        return documents
