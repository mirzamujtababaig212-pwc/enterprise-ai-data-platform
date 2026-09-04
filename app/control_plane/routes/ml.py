from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, HTTPException

from app.control_plane.dependencies import get_vehicle_risk_predictor
from ml.api.vehicle_risk import (
    VehicleRiskRequest,
    VehicleRiskResponse,
)

router = APIRouter(
    prefix="/api/v1/ml",
    tags=["machine-learning"],
)


@router.post(
    "/vehicle-risk/predict",
    response_model=VehicleRiskResponse,
)
def predict_vehicle_risk(
    request: VehicleRiskRequest,
) -> VehicleRiskResponse:
    predictor = get_vehicle_risk_predictor()

    try:
        dataframe = pd.DataFrame([request.model_dump()])
        prediction = predictor.predict(dataframe)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    return VehicleRiskResponse(
        risk=prediction.risk,
        risk_probability=prediction.risk_probability,
        model_name=prediction.model_name,
        model_alias=prediction.model_alias,
    )
