"""Small FastAPI service. Run: MODEL_DIR=models/bottle uvicorn industrial_anomaly.api:app"""

from __future__ import annotations

import base64
import os
import threading
import time
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel

from .detector import AnomalyDetector

MAX_BYTES = 10 * 1024 * 1024
_state: dict = {"detector": None, "lock": threading.Lock()}


class PredictResponse(BaseModel):
    score: float
    threshold: float | None
    verdict: str
    latency_ms: float
    overlay_png_base64: str | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    model_dir = os.environ.get("MODEL_DIR")
    if model_dir:
        _state["detector"] = AnomalyDetector.load(model_dir, os.environ.get("DEVICE"))
    yield


app = FastAPI(title="Industrial Anomaly Detection", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _state["detector"] is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(file: UploadFile, return_overlay: bool = False) -> PredictResponse:
    det: AnomalyDetector | None = _state["detector"]
    if det is None:
        raise HTTPException(503, "Model not loaded; set MODEL_DIR.")
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image too large (max 10 MB).")
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(400, "Could not decode image.")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    t0 = time.perf_counter()
    with _state["lock"]:
        pred = det.predict_arrays([rgb])[0]
    latency = (time.perf_counter() - t0) * 1000

    verdict = "UNKNOWN" if pred.is_anomalous is None else ("FAIL" if pred.is_anomalous else "PASS")
    overlay_b64 = None
    if return_overlay:
        viz = cv2.cvtColor(det.render(rgb, pred), cv2.COLOR_RGB2BGR)
        overlay_b64 = base64.b64encode(cv2.imencode(".png", viz)[1].tobytes()).decode()
    return PredictResponse(
        score=pred.score,
        threshold=det.threshold,
        verdict=verdict,
        latency_ms=round(latency, 2),
        overlay_png_base64=overlay_b64,
    )
