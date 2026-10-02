import numpy as np
import pytest
from fastapi.testclient import TestClient

from industrial_anomaly import api
from industrial_anomaly.data import list_images


@pytest.fixture()
def client(fitted):
    api._state["detector"] = fitted
    with TestClient(api.app) as c:  # lifespan without MODEL_DIR leaves the injected detector
        yield c
    api._state["detector"] = None


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_predict_returns_verdict_and_optional_overlay(client, mvtec_root):
    path = list_images(mvtec_root / "bottle/test/broken")[0]
    r = client.post(
        "/predict?return_overlay=true", files={"file": ("x.png", path.read_bytes(), "image/png")}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["verdict"] in {"PASS", "FAIL"} and body["score"] > 0
    assert body["overlay_png_base64"]


def test_predict_rejects_garbage(client):
    r = client.post("/predict", files={"file": ("x.png", b"not an image", "image/png")})
    assert r.status_code == 400


def test_predict_without_model_is_503():
    api._state["detector"] = None
    with TestClient(api.app) as c:
        r = c.post(
            "/predict", files={"file": ("x.png", np.zeros(10, np.uint8).tobytes(), "image/png")}
        )
    assert r.status_code == 503
