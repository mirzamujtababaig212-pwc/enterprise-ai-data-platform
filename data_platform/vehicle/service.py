from __future__ import annotations

from datetime import datetime
from typing import Any

from common.config.settings import Settings
from common.readers.delta_reader import DeltaReader
from common.spark.spark_builder import SparkSessionBuilder


class VehicleDataService:
    """Bounded read-only access to canonical vehicle telemetry data."""

    SOURCE_TABLE = "silver.vehicle_events"

    def __init__(
        self,
        *,
        reader: DeltaReader | None = None,
        spark: Any | None = None,
    ) -> None:
        self._reader = reader or DeltaReader(
            path=Settings.storage.SILVER_PATH,
        )
        self._spark = spark

    def query(
        self,
        *,
        vehicle_id: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        self._validate(
            vehicle_id=vehicle_id,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
        )

        spark = self._spark or SparkSessionBuilder.build(
            app_name="EnterpriseAIVehicleDataQuery",
        )

        dataframe = self._reader.read(spark).select(
            "vehicle_id",
            "event_time",
            "speed",
        )

        if vehicle_id is not None:
            dataframe = dataframe.where(
                dataframe.vehicle_id == vehicle_id.strip(),
            )

        if start_time is not None:
            dataframe = dataframe.where(
                dataframe.event_time >= start_time,
            )

        if end_time is not None:
            dataframe = dataframe.where(
                dataframe.event_time <= end_time,
            )

        dataframe = dataframe.orderBy(
            dataframe.event_time.asc(),
            dataframe.vehicle_id.asc(),
        )

        rows = dataframe.limit(limit).collect()

        records = [
            {
                "vehicle_id": row["vehicle_id"],
                "event_time": (
                    row["event_time"].isoformat() if row["event_time"] is not None else None
                ),
                "speed": row["speed"],
            }
            for row in rows
        ]

        return {
            "source": self.SOURCE_TABLE,
            "filters": {
                "vehicle_id": (vehicle_id.strip() if vehicle_id is not None else None),
                "start_time": (start_time.isoformat() if start_time is not None else None),
                "end_time": (end_time.isoformat() if end_time is not None else None),
                "limit": limit,
            },
            "count": len(records),
            "records": records,
        }

    @staticmethod
    def _validate(
        *,
        vehicle_id: str | None,
        start_time: datetime | None,
        end_time: datetime | None,
        limit: int,
    ) -> None:
        if vehicle_id is not None:
            if not isinstance(vehicle_id, str):
                raise TypeError("vehicle_id must be a string.")

            if not vehicle_id.strip():
                raise ValueError("vehicle_id must not be empty.")

        if start_time is not None and not isinstance(start_time, datetime):
            raise TypeError("start_time must be a datetime.")

        if end_time is not None and not isinstance(end_time, datetime):
            raise TypeError("end_time must be a datetime.")

        if start_time is not None and end_time is not None and start_time > end_time:
            raise ValueError("start_time must be before or equal to end_time.")

        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TypeError("limit must be an integer.")

        if limit < 1 or limit > 100:
            raise ValueError("limit must be between 1 and 100.")
