"""Production FastAPI service — registry, monitoring and drift detection.

This is the Lecture 9 service grown up: the model now comes from a versioned
registry, every prediction feeds a monitor, and drift is reported without needing
any labels.

    cd lectures/Lecture_16/app
    python seed_registry.py       # train two versions and register them
    uvicorn main:app --reload     # http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parents[2]
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(ROOT / "lectures" / "Lecture_09" / "app"))

from model import ModelBundle, PreprocessConfig, ShapeClassifier   # noqa: E402
from monitoring import Monitor                                      # noqa: E402
from registry import ModelRegistry                                  # noqa: E402

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
ALLOWED_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/bmp", "image/webp"}

STATE: dict = {"bundle": None, "record": None, "monitor": None}
registry = ModelRegistry()


def load_version(version: str) -> None:
    """Load a registered version into the serving slot, and reset the monitor."""
    payload = registry.load_payload(version)
    record = registry.get(version)

    model = ShapeClassifier(n_classes=len(payload["classes"]))
    model.load_state_dict(payload["state_dict"])
    bundle = ModelBundle(
        model=model,
        classes=payload["classes"],
        preprocess=PreprocessConfig(**payload["preprocess"]),
        version=version,
        metrics=record["metrics"],
    )
    bundle.warmup()

    STATE["bundle"] = bundle
    STATE["record"] = record
    # A new model means a new baseline; carrying the old one over would produce
    # a drift alert that is really just a deployment.
    STATE["monitor"] = Monitor(payload["classes"], record.get("baseline_distribution"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    torch.set_num_threads(1)
    prod = registry.production()
    if prod:
        load_version(prod["version"])
        print(f"serving version {prod['version']}")
    else:
        print("WARNING: no production model registered. Run seed_registry.py.")
    yield
    STATE["bundle"] = None


app = FastAPI(
    title="Shape Classifier — Production Service",
    description="Deep Learning course, Lecture 16 — registry, monitoring, drift detection.",
    version="2.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------
class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]
    model_version: str
    inference_ms: float
    model_config = {"protected_namespaces": ()}


class ModelInfo(BaseModel):
    version: str
    stage: str
    created_at: str
    metrics: dict
    notes: str


class PromoteResponse(BaseModel):
    promoted: str
    previous: str | None
    now_serving: str


def get_bundle() -> ModelBundle:
    if STATE["bundle"] is None:
        raise HTTPException(status_code=503,
                            detail="No model loaded. Run seed_registry.py and restart.")
    return STATE["bundle"]


async def read_validated(file: UploadFile) -> bytes:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415,
                            detail=f"Unsupported content type {file.content_type!r}.")
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large.")
    if not raw:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")
    return raw


# --------------------------------------------------------------------------
@app.get("/health", tags=["ops"])
def health():
    bundle = STATE["bundle"]
    monitor = STATE["monitor"]
    return {
        "status": "ok" if bundle else "degraded",
        "model_loaded": bundle is not None,
        "model_version": bundle.version if bundle else None,
        "uptime_seconds": round(monitor.snapshot()["uptime_seconds"], 2) if monitor else 0,
    }


@app.get("/metrics", tags=["ops"])
def metrics():
    """Operational metrics plus the drift report.

    `drift.psi` is the number to alert on: it needs no labels, and it moves
    before accuracy does.
    """
    if STATE["monitor"] is None:
        raise HTTPException(status_code=503, detail="No model loaded.")
    return STATE["monitor"].snapshot()


@app.get("/models", response_model=list[ModelInfo], tags=["registry"])
def list_models():
    return [ModelInfo(version=r["version"], stage=r["stage"], created_at=r["created_at"],
                      metrics=r["metrics"], notes=r["notes"])
            for r in registry.list_models()]


@app.post("/models/{version}/promote", response_model=PromoteResponse, tags=["registry"])
def promote(version: str):
    """Promote a version to production and start serving it immediately.

    This is also the rollback path: promoting the previous version restores it.
    """
    previous = STATE["bundle"].version if STATE["bundle"] else None
    try:
        registry.promote(version)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown version {version}.")
    load_version(version)
    return PromoteResponse(promoted=version, previous=previous, now_serving=version)


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
async def predict(file: UploadFile = File(...)):
    bundle = get_bundle()
    monitor = STATE["monitor"]
    t0 = time.perf_counter()

    raw = await read_validated(file)
    try:
        tensor = bundle.bytes_to_tensor(raw)
    except ValueError as e:
        monitor.record_error()
        raise HTTPException(status_code=422, detail=str(e)) from e

    probs, infer_ms = bundle.predict(tensor)
    label, confidence = bundle.top_prediction(probs[0])
    monitor.record(bundle.classes.index(label), confidence,
                   (time.perf_counter() - t0) * 1000)

    return PredictionResponse(
        predicted_class=label,
        confidence=round(confidence, 6),
        probabilities=bundle.as_dict(probs[0]),
        model_version=bundle.version,
        inference_ms=round(infer_ms, 3),
    )


@app.get("/model-card", tags=["ops"])
def model_card():
    """The model card ships with the service, not in a forgotten wiki page."""
    path = APP_DIR / "MODEL_CARD.md"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No model card written yet.")
    return {"markdown": path.read_text(encoding="utf-8")}


@app.get("/", tags=["ops"])
def root():
    return {
        "service": "Shape Classifier — Production",
        "docs": "/docs",
        "endpoints": ["/health", "/metrics", "/models", "/models/{version}/promote",
                      "/predict", "/model-card"],
    }
