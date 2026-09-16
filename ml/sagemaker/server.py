from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ml.sagemaker.vehicle_risk import input_fn, model_fn, output_fn, predict_fn

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/opt/ml/model"))

app = FastAPI(
    title="Vehicle Risk SageMaker Inference",
    version="1.0.0",
)

MODEL: Any | None = None


def initialize_model() -> Any:
    """Load the packaged Vehicle Risk model into the inference process."""
    global MODEL

    MODEL = model_fn(MODEL_DIR)
    return MODEL


@app.get("/ping")
def ping() -> JSONResponse:
    """SageMaker health-check endpoint."""
    if MODEL is None:
        return JSONResponse(
            status_code=503,
            content={"detail": "model is not loaded"},
        )

    return JSONResponse(
        status_code=200,
        content={"status": "ok"},
    )


@app.post("/invocations")
async def invocations(request: Request) -> JSONResponse:
    """SageMaker inference endpoint."""
    if MODEL is None:
        return JSONResponse(
            status_code=503,
            content={"detail": "model is not loaded"},
        )

    content_type = request.headers.get("content-type", "")
    accept = request.headers.get("accept", "application/json")
    request_body = await request.body()

    try:
        input_data = input_fn(request_body, content_type)
        prediction = predict_fn(input_data, MODEL)
        response_body = output_fn(prediction, accept)
    except ValueError as exc:
        return JSONResponse(
            status_code=400,
            content={"detail": str(exc)},
        )

    return JSONResponse(
        status_code=200,
        content=json.loads(response_body),
    )


def main() -> None:
    """Load the model and start the SageMaker inference server."""
    initialize_model()

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8080")),
    )


if __name__ == "__main__":
    main()
