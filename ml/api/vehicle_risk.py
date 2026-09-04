from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class VehicleRiskRequest(BaseModel):
    """
    Request payload for vehicle-risk inference.
    """

    model_config = ConfigDict(extra="forbid")

    event_count: int = Field(ge=0)

    avg_speed: float
    max_speed: float
    speed_stddev: float

    avg_rpm: float
    max_rpm: float

    avg_fuel_level: float
    min_fuel_level: float

    avg_battery: float

    avg_engine_temperature: float
    max_engine_temperature: float


class VehicleRiskResponse(BaseModel):
    """
    Response returned by vehicle-risk inference.
    """

    risk: int = Field(
        ge=0,
        le=1,
    )

    risk_probability: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    model_name: str

    model_alias: str


__all__ = [
    "VehicleRiskRequest",
    "VehicleRiskResponse",
]
