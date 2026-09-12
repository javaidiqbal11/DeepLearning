"""Tests for the production service: registry, promotion, rollback and drift.

    cd lectures/Lecture_16/app
    python seed_registry.py
    pytest -v
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

from main import app, registry          # noqa: E402
from monitoring import population_stability_index, psi_status   # noqa: E402

needs_registry = pytest.mark.skipif(
    registry.production() is None,
    reason="run `python seed_registry.py` first",
)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def png(colour=(210, 70, 70), size=(32, 32)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, colour).save(buf, format="PNG")
    return buf.getvalue()


def sample(name="circle"):
    p = APP_DIR.parents[1] / "Lecture_09" / "app" / "samples" / f"{name}.png"
    return p.read_bytes() if p.exists() else png()


# --------------------------------------------------------------------------
# PSI is the load-bearing piece of the monitoring story, so test it directly.
# --------------------------------------------------------------------------
def test_psi_zero_for_identical_distributions():
    d = [0.25, 0.25, 0.25, 0.25]
    assert population_stability_index(d, d) == pytest.approx(0.0, abs=1e-9)


def test_psi_grows_with_divergence():
    base = [0.25, 0.25, 0.25, 0.25]
    mild = [0.30, 0.25, 0.23, 0.22]
    severe = [0.90, 0.04, 0.03, 0.03]
    assert population_stability_index(base, mild) < population_stability_index(base, severe)


def test_psi_is_symmetric():
    a, b = [0.4, 0.3, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4]
    assert population_stability_index(a, b) == pytest.approx(
        population_stability_index(b, a), rel=1e-9)


def test_psi_status_thresholds():
    assert psi_status(0.05) == "stable"
    assert psi_status(0.15) == "investigate"
    assert psi_status(0.40) == "alert"


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------
@needs_registry
def test_lists_registered_models(client):
    r = client.get("/models")
    assert r.status_code == 200
    versions = {m["version"] for m in r.json()}
    assert {"0.9.0", "1.0.0"} <= versions
    in_production = [m for m in r.json() if m["stage"] == "production"]
    assert len(in_production) == 1, "exactly one version may be in production"


@needs_registry
def test_promotion_and_rollback(client):
    original = client.get("/health").json()["model_version"]

    # Roll back to the older version...
    r = client.post("/models/0.9.0/promote")
    assert r.status_code == 200
    assert r.json()["now_serving"] == "0.9.0"
    assert client.get("/health").json()["model_version"] == "0.9.0"

    # ...and forward again. Rollback is just promotion in the other direction.
    r = client.post(f"/models/{original}/promote")
    assert r.status_code == 200
    assert client.get("/health").json()["model_version"] == original


@needs_registry
def test_promoting_unknown_version_is_404(client):
    assert client.post("/models/9.9.9/promote").status_code == 404


def test_versions_are_immutable():
    """Re-registering a version must fail. Silent overwrite destroys traceability."""
    if registry.get("1.0.0") is None:
        pytest.skip("registry not seeded")
    with pytest.raises(ValueError):
        registry.register({"state_dict": {}, "classes": [], "preprocess": {}},
                          "1.0.0", {}, {})


# --------------------------------------------------------------------------
# inference and monitoring
# --------------------------------------------------------------------------
@needs_registry
def test_predict(client):
    r = client.post("/predict", files={"file": ("c.png", sample(), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_class"] in ["circle", "square", "triangle", "star"]
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-4


@needs_registry
def test_metrics_report_drift(client):
    for _ in range(20):
        client.post("/predict", files={"file": ("c.png", sample(), "image/png")})

    m = client.get("/metrics").json()
    assert m["requests_total"] >= 20
    assert m["latency_p50_ms"] is not None
    drift = m["drift"]
    assert "psi" in drift and "status" in drift
    assert drift["status"] in {"stable", "investigate", "alert"}


@needs_registry
def test_single_class_traffic_raises_psi(client):
    """Feeding only one class is a distribution shift, and PSI must notice."""
    client.post("/models/1.0.0/promote")          # reset the monitor
    for _ in range(60):
        client.post("/predict", files={"file": ("c.png", sample("circle"), "image/png")})

    drift = client.get("/metrics").json()["drift"]
    assert drift["psi"] > 0.25, (
        f"PSI {drift['psi']} did not alert on single-class traffic")
    assert drift["status"] == "alert"


@needs_registry
def test_errors_are_counted(client):
    before = client.get("/metrics").json()["errors_total"]
    client.post("/predict", files={"file": ("x.png", b"\x89PNG\r\n\x1a\ngarbage", "image/png")})
    after = client.get("/metrics").json()["errors_total"]
    assert after == before + 1
