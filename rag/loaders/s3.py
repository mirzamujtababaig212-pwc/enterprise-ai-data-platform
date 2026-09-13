from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from pyspark.sql import SparkSession

from common.provenance import s3_source_ref
from common.readers.base_reader import BaseReader
from rag.loaders.spark import SparkDataFrameDocumentLoader
from rag.models import Document


class S3DocumentLoader:
    """
    Convert an S3-backed Spark dataset into canonical RAG Documents.

    The loader composes:
    - an existing BaseReader for physical data access
    - SparkDataFrameDocumentLoader for row-to-Document conversion
    - the canonical S3 provenance adapter for source identity

    It deliberately does not own RAG indexing, chunking, embeddings,
    or vector-store behavior.
    """

    def __init__(
        self,
        *,
        reader: BaseReader,
        path: str,
        id_fn: Callable[[dict[str, Any]], str],
        content_fn: Callable[[dict[str, Any]], str],
        metadata_fn: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> None:
        self._reader = reader
        self._source_ref = s3_source_ref(path)
        self._document_loader = SparkDataFrameDocumentLoader(
            id_fn=id_fn,
            content_fn=content_fn,
            metadata_fn=metadata_fn,
        )

    def load(
        self,
        spark: SparkSession,
    ) -> Sequence[Document]:
        dataframe = self._reader.read(spark)

        documents = self._document_loader.load(dataframe)

        source_ref = self._source_ref.to_dict()

        return [
            Document(
                id=document.id,
                content=document.content,
                metadata={
                    **document.metadata,
                    "source_ref": dict(source_ref),
                },
            )
            for document in documents
        ]
