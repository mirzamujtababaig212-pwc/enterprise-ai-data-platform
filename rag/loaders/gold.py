from __future__ import annotations

from collections.abc import Sequence

from pyspark.sql import SparkSession

from common.readers.base_reader import BaseReader
from rag.loaders.vehicle_metrics import VehicleMetricsDocumentMapper
from rag.models import Document


class GoldVehicleMetricsDocumentLoader:
    """
    Load the canonical Gold vehicle_metrics dataset into RAG Documents.

    The adapter composes:
      1. an existing Spark-compatible reader
      2. the existing vehicle metrics document mapper

    It deliberately does not contain indexing, embedding, retrieval,
    or vector-store logic.
    """

    def __init__(
        self,
        reader: BaseReader,
    ) -> None:
        self._reader = reader
        self._document_loader = VehicleMetricsDocumentMapper.loader()

    def load(
        self,
        spark: SparkSession,
    ) -> Sequence[Document]:
        dataframe = self._reader.read(spark)
        return self._document_loader.load(dataframe)
