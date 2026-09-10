from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from pyspark.sql import SparkSession

from common.readers.base_reader import BaseReader
from rag.dbt.models import DbtModel
from rag.loaders.spark import SparkDataFrameDocumentLoader
from rag.models import Document


class DbtModelDocumentLoader:
    """
    Convert a dbt model's queryable data into canonical RAG Documents.

    The loader composes:
    - a dbt model for semantic metadata
    - an existing BaseReader for physical data access
    - SparkDataFrameDocumentLoader for row-to-Document conversion

    It deliberately does not own RAG indexing, chunking, embeddings,
    or vector-store behavior.
    """

    def __init__(
        self,
        *,
        model: DbtModel,
        reader: BaseReader,
        id_fn: Callable[[dict[str, Any]], str],
        content_fn: Callable[[dict[str, Any]], str],
        metadata_fn: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> None:
        self._model = model
        self._reader = reader
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

        return [
            Document(
                id=document.id,
                content=document.content,
                metadata={
                    **document.metadata,
                    "dbt_unique_id": self._model.unique_id,
                    "dbt_model": self._model.name,
                    "dbt_database": self._model.database,
                    "dbt_schema": self._model.schema,
                    "dbt_alias": self._model.alias,
                    "dbt_materialization": self._model.materialization,
                },
            )
            for document in documents
        ]
