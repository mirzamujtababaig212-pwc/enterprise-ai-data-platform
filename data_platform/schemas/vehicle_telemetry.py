from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class VehicleTelemetry(BaseModel):
    vehicle_id: str = Field(min_length=1)
    event_time: datetime

    latitude: float = Field(ge=-90.0, le=90.0)
    longitude: float = Field(ge=-180.0, le=180.0)

    speed: float = Field(ge=0.0)
    rpm: int = Field(ge=0)
    fuel_level: float = Field(ge=0.0, le=100.0)
    battery: float = Field(ge=0.0, le=100.0)
    engine_temperature: float
    gear: int = Field(ge=0)

    @field_validator("vehicle_id")
    @classmethod
    def normalize_vehicle_id(cls, value: str) -> str:
        return value.strip().upper()

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
