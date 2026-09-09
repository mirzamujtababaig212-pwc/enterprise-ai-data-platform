from __future__ import annotations

from datetime import datetime
from typing import Any

from rag.loaders.spark import SparkDataFrameDocumentLoader


class VehicleMetricsDocumentMapper:
    """
    Map Gold vehicle_metrics rows into canonical RAG Documents.

    Gold grain:
        one row per vehicle_id

    Document grain:
        one Document per vehicle_id
    """

    REQUIRED_COLUMNS = (
        "vehicle_id",
        "event_count",
        "avg_speed",
        "min_speed",
        "max_speed",
        "first_event_time",
        "last_event_time",
    )

    @staticmethod
    def document_id(row: dict[str, Any]) -> str:
        vehicle_id = row.get("vehicle_id")

        if vehicle_id is None or not str(vehicle_id).strip():
            raise ValueError("vehicle_id must be non-empty")

        return f"vehicle:{vehicle_id}"

    @staticmethod
    def content(row: dict[str, Any]) -> str:
        vehicle_id = row["vehicle_id"]

        return (
            f"Vehicle {vehicle_id} recorded {row['event_count']} events. "
            f"Average speed was {float(row['avg_speed']):.2f}. "
            f"Minimum speed was {float(row['min_speed']):.2f}. "
            f"Maximum speed was {float(row['max_speed']):.2f}. "
            f"The observation period started at "
            f"{VehicleMetricsDocumentMapper._format_timestamp(row['first_event_time'])} "
            f"and ended at "
            f"{VehicleMetricsDocumentMapper._format_timestamp(row['last_event_time'])}."
        )

    @staticmethod
    def metadata(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "source": "gold.vehicle_metrics",
            "data_layer": "gold",
            "dataset": "vehicle_metrics",
            "entity_type": "vehicle",
            "vehicle_id": str(row["vehicle_id"]),
        }

    @staticmethod
    def loader() -> SparkDataFrameDocumentLoader:
        return SparkDataFrameDocumentLoader(
            id_fn=VehicleMetricsDocumentMapper.document_id,
            content_fn=VehicleMetricsDocumentMapper.content,
            metadata_fn=VehicleMetricsDocumentMapper.metadata,
        )

    @staticmethod
    def _format_timestamp(value: Any) -> str:
        if isinstance(value, datetime):
            return value.isoformat()

        return str(value)
