"""Generic MLflow PyFunc inference server for any CV project adapter."""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Any

import mlflow
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel

REQUESTS = Counter("mlforge_inference_requests_total", "Inference requests", ["status"])
LATENCY = Histogram("mlforge_inference_latency_seconds", "Inference latency")
MODEL: Any = None


class PredictionRequest(BaseModel):
    inputs: Any


@asynccontextmanager
async def lifespan(_: FastAPI):
    global MODEL
    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    model_uri = os.environ["MODEL_URI"]
    mlflow.set_tracking_uri(tracking_uri)
    MODEL = mlflow.pyfunc.load_model(model_uri)
    yield


app = FastAPI(title="MLForge CV model server", lifespan=lifespan)
app.mount("/metrics", make_asgi_app())


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model_uri": os.environ.get("MODEL_URI", "")}


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, Any]:
    if MODEL is None:
        REQUESTS.labels(status="not_ready").inc()
        raise HTTPException(status_code=503, detail="model is loading")
    started = time.monotonic()
    try:
        prediction = MODEL.predict(request.inputs)
        result = prediction.tolist() if hasattr(prediction, "tolist") else prediction
        REQUESTS.labels(status="success").inc()
        return {"predictions": result, "model_uri": os.environ["MODEL_URI"]}
    except Exception as exc:
        REQUESTS.labels(status="error").inc()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        LATENCY.observe(time.monotonic() - started)
