"""Small portable Canary server for an MLflow-tracked YOLO candidate.

This is deliberately not branded as Triton: NVIDIA Triton is an x86/NVIDIA
runtime and is not a reliable local target on Apple Silicon k3d.  The HTTP,
metrics, alias-resolution, health, and rollback contract is the same one the
real Triton deployment will use in a GPU cluster.
"""
from __future__ import annotations

import base64
import io
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import mlflow
from fastapi import FastAPI, HTTPException
from PIL import Image
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel, Field
from ultralytics import YOLO

REQUESTS = Counter("mlforge_yolo_canary_requests_total", "YOLO Canary requests", ["status"])
LATENCY = Histogram("mlforge_yolo_canary_latency_seconds", "YOLO Canary request latency")
MODEL: YOLO | None = None
MODEL_VERSION = ""


class PredictionRequest(BaseModel):
    image_b64: str = Field(description="Base64-encoded JPEG or PNG image")
    confidence: float = Field(default=0.25, ge=0.0, le=1.0)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global MODEL, MODEL_VERSION
    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    model_name = os.environ["MODEL_NAME"]
    model_alias = os.environ.get("MODEL_ALIAS", "demo-champion")
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()
    version = client.get_model_version_by_alias(model_name, model_alias)
    weights = mlflow.artifacts.download_artifacts(
        run_id=version.run_id, artifact_path="weights/best.pt", dst_path="/tmp/model"
    )
    MODEL = YOLO(weights)
    MODEL_VERSION = str(version.version)
    yield


app = FastAPI(title="MLForge local YOLO Canary", lifespan=lifespan)
app.mount("/metrics", make_asgi_app())


@app.get("/health")
def health() -> dict[str, str]:
    if MODEL is None:
        raise HTTPException(status_code=503, detail="model is loading")
    return {
        "status": "ok",
        "model_name": os.environ["MODEL_NAME"],
        "model_alias": os.environ.get("MODEL_ALIAS", "demo-champion"),
        "model_version": MODEL_VERSION,
    }


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, Any]:
    if MODEL is None:
        REQUESTS.labels(status="not_ready").inc()
        raise HTTPException(status_code=503, detail="model is loading")
    started = time.monotonic()
    try:
        image = Image.open(io.BytesIO(base64.b64decode(request.image_b64))).convert("RGB")
        result = MODEL.predict(image, conf=request.confidence, verbose=False)[0]
        names = result.names
        detections = [
            {
                "class_id": int(box.cls.item()),
                "class_name": str(names[int(box.cls.item())]),
                "confidence": round(float(box.conf.item()), 6),
                "xyxy": [round(float(value), 2) for value in box.xyxy[0].tolist()],
            }
            for box in result.boxes
        ]
        REQUESTS.labels(status="success").inc()
        return {"detections": detections, "model_version": MODEL_VERSION}
    except Exception as exc:
        REQUESTS.labels(status="error").inc()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        LATENCY.observe(time.monotonic() - started)
