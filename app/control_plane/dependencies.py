from __future__ import annotations

from ai_platform.llm_gateway.routing.router import Router
from ml.inference import VehicleRiskPredictor

_llm_router = Router()
_vehicle_risk_predictor = VehicleRiskPredictor(
    model_alias="champion",
)


def get_llm_router() -> Router:
    return _llm_router


def get_vehicle_risk_predictor() -> VehicleRiskPredictor:
    return _vehicle_risk_predictor
