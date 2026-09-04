from __future__ import annotations

import pandas as pd
from fastapi import APIRouter

from ml.inference import VehicleRiskPredictor

from .vehicle_risk import (
    VehicleRiskRequest,
    VehicleRiskResponse,
)


router = APIRouter(
    prefix="/api/v1/ml",
    tags=["machine-learning"],
)


_predictor = VehicleRiskPredictor(model_alias="champion")


@router.post(
    "/vehicle-risk/predict",
    response_model=VehicleRiskResponse,
)
def predict_vehicle_risk(
    request: VehicleRiskRequest,
) -> VehicleRiskResponse:

    dataframe = pd.DataFrame([request.model_dump()])

    prediction = _predictor.predict(dataframe)

    return VehicleRiskResponse(
        risk=prediction.risk,
        risk_probability=(prediction.risk_probability),
        model_name=prediction.model_name,
        model_alias=prediction.model_alias,
    )
