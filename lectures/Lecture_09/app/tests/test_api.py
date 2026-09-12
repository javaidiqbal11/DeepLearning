"""Endpoint tests for the shape classifier service.

    cd lectures/Lecture_09/app
    pytest -v

These test the *contract* — status codes and response shape — separately from
model quality. A service can be perfectly correct and still serve a bad model,
and the two failures need different fixes.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))

from main import app  # noqa: E402

SAMPLES = APP_DIR / "samples"
ARTEFACT = APP_DIR / "artefacts" / "shape_classifier.pt"

needs_artefact = pytest.mark.skipif(
    not ARTEFACT.exists(),
    reason="run `python train_and_export.py` first to create the model artefact",
)


@pytest.fixture(scope="module")
def client():
    # The context manager form is required: it runs the lifespan handler, which
    # is what loads the model.
    with TestClient(app) as c:
        yield c


def png_bytes(colour=(200, 60, 60), size=(32, 32)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, colour).save(buf, format="PNG")
    return buf.getvalue()


def sample_file(name="circle.png"):
    return (SAMPLES / name).read_bytes()


# --------------------------------------------------------------------------
# operational endpoints
# --------------------------------------------------------------------------
def test_root_lists_endpoints(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "/predict" in r.json()["endpoints"]


def test_health_reports_model_state(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in {"ok", "degraded"}
    assert "uptime_seconds" in body
    if body["model_loaded"]:
        assert body["classes"] == ["circle", "square", "triangle", "star"]


@needs_artefact
def test_metrics_shape(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    for key in ("requests_total", "errors_total", "prediction_counts"):
        assert key in r.json()


# --------------------------------------------------------------------------
# the happy path
# --------------------------------------------------------------------------
@needs_artefact
def test_predict_returns_full_schema(client):
    r = client.post("/predict", files={"file": ("circle.png", sample_file(), "image/png")})
    assert r.status_code == 200
    body = r.json()

    assert body["predicted_class"] in ["circle", "square", "triangle", "star"]
    assert 0.0 <= body["confidence"] <= 1.0
    assert len(body["probabilities"]) == 4
    assert body["inference_ms"] > 0
    assert body["model_version"] == "1.0.0"
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-4


@needs_artefact
@pytest.mark.parametrize("name", ["circle", "square", "triangle", "star"])
def test_golden_predictions(client, name):
    """Golden test: a known image must keep producing a known answer.

    This is what catches a preprocessing change that silently breaks the model
    without breaking any endpoint.
    """
    r = client.post("/predict", files={"file": (f"{name}.png", sample_file(f"{name}.png"),
                                                "image/png")})
    assert r.status_code == 200
    assert r.json()["predicted_class"] == name, (
        f"expected {name}, got {r.json()['predicted_class']} — "
        "the model or the preprocessing has changed"
    )


@needs_artefact
def test_batch_endpoint(client):
    files = [("files", (f"{n}.png", sample_file(f"{n}.png"), "image/png"))
             for n in ("circle", "square", "triangle")]
    r = client.post("/predict/batch", files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 3
    assert len(body["predictions"]) == 3
    assert body["ms_per_image"] > 0


@needs_artefact
def test_accepts_non_native_size(client):
    """A 128x128 upload must be resized, not rejected."""
    r = client.post("/predict",
                    files={"file": ("big.png", png_bytes(size=(128, 128)), "image/png")})
    assert r.status_code == 200


# --------------------------------------------------------------------------
# error handling
# --------------------------------------------------------------------------
def test_rejects_wrong_content_type(client):
    r = client.post("/predict",
                    files={"file": ("notes.txt", b"this is not an image", "text/plain")})
    assert r.status_code == 415
    assert "content type" in r.json()["detail"].lower()


def test_rejects_oversized_file(client):
    big = b"\x89PNG\r\n\x1a\n" + b"0" * (6 * 1024 * 1024)
    r = client.post("/predict", files={"file": ("big.png", big, "image/png")})
    assert r.status_code == 413


@needs_artefact
def test_rejects_corrupt_image(client):
    """Correct content type, undecodable bytes -> 422, not a 500."""
    r = client.post("/predict",
                    files={"file": ("broken.png", b"\x89PNG\r\n\x1a\ngarbage", "image/png")})
    assert r.status_code == 422
    assert "decode" in r.json()["detail"].lower()


def test_rejects_empty_file(client):
    r = client.post("/predict", files={"file": ("empty.png", b"", "image/png")})
    assert r.status_code == 422


def test_missing_file_is_422(client):
    """FastAPI's own validation handles a missing required field."""
    assert client.post("/predict").status_code == 422


@needs_artefact
def test_batch_over_limit_rejected(client):
    files = [("files", (f"{i}.png", png_bytes(), "image/png")) for i in range(40)]
    r = client.post("/predict/batch", files=files)
    assert r.status_code == 413


# --------------------------------------------------------------------------
# preprocessing contract
# --------------------------------------------------------------------------
@needs_artefact
def test_preprocessing_is_deterministic():
    """The same bytes must always produce the same tensor."""
    from model import ModelBundle
    bundle = ModelBundle.load()
    raw = sample_file()
    a = bundle.bytes_to_tensor(raw)
    b = bundle.bytes_to_tensor(raw)
    assert (a == b).all()
    assert a.shape == (1, 3, 32, 32)


@needs_artefact
def test_artefact_carries_its_preprocessing():
    """A checkpoint without its preprocessing config is not deployable."""
    from model import ModelBundle
    bundle = ModelBundle.load()
    assert bundle.preprocess.input_size == 32
    assert len(bundle.preprocess.mean) == 3
    assert len(bundle.preprocess.std) == 3
    assert bundle.version
    assert "test_accuracy" in bundle.metrics
