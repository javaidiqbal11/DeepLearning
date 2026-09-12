"""FastAPI service for the shape classifier.

Run it:
    cd lectures/Lecture_09/app
    python train_and_export.py          # once, to create the artefact
    uvicorn main:app --reload

Then open http://127.0.0.1:8000/docs for the generated documentation.
"""
from __future__ import annotations

import statistics
import time
from collections import Counter, deque
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from model import ModelBundle
from schemas import (BatchItem, BatchPredictionResponse, HealthResponse,
                     MetricsResponse, PredictionResponse)

MAX_UPLOAD_BYTES = 5 * 1024 * 1024          # 5 MB
MAX_BATCH_SIZE = 32
ALLOWED_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/bmp", "image/webp"}

# Process-wide state, populated at startup.
STATE: dict = {"bundle": None, "started_at": None}
METRICS: dict = {"requests": 0, "errors": 0,
                 "latencies": deque(maxlen=1000),
                 "predictions": Counter()}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once, at startup — never per request."""
    # Thread oversubscription is a common and invisible latency cliff: each
    # request already runs in its own worker, so let each use a single thread.
    torch.set_num_threads(1)

    STATE["started_at"] = time.time()
    try:
        bundle = ModelBundle.load()
        bundle.warmup()
        STATE["bundle"] = bundle
        print(f"model loaded: version {bundle.version}, classes {bundle.classes}")
    except FileNotFoundError as e:
        # Start anyway so /health can report the problem instead of the process
        # crash-looping under an orchestrator.
        print(f"WARNING: {e}")
    yield
    STATE["bundle"] = None


app = FastAPI(
    title="Shape Classifier API",
    description="Deep Learning course, Lecture 9 — serving a trained CNN with FastAPI.",
    version="1.0.0",
    lifespan=lifespan,
)


def get_bundle() -> ModelBundle:
    bundle = STATE["bundle"]
    if bundle is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Run train_and_export.py and restart the service.",
        )
    return bundle


async def read_validated(file: UploadFile) -> bytes:
    """Read an upload, enforcing content type and size limits."""
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported content type {file.content_type!r}. "
                   f"Allowed: {sorted(ALLOWED_TYPES)}",
        )
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File is {len(raw)} bytes; the limit is {MAX_UPLOAD_BYTES}.",
        )
    if not raw:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded file is empty.",
        )
    return raw


# --------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health():
    """Liveness and readiness. Orchestrators poll this."""
    bundle = STATE["bundle"]
    return HealthResponse(
        status="ok" if bundle else "degraded",
        model_loaded=bundle is not None,
        model_version=bundle.version if bundle else None,
        classes=bundle.classes if bundle else None,
        uptime_seconds=round(time.time() - STATE["started_at"], 2),
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["ops"])
def metrics():
    """Latency percentiles and the prediction distribution.

    The prediction distribution is the cheapest drift signal available: it needs
    no labels, and a sudden change in it usually precedes a measurable accuracy
    drop. Lecture 16 builds this out properly.
    """
    lat = sorted(METRICS["latencies"])

    def pct(p):
        if not lat:
            return None
        return round(lat[min(int(len(lat) * p), len(lat) - 1)], 3)

    return MetricsResponse(
        requests_total=METRICS["requests"],
        errors_total=METRICS["errors"],
        latency_p50_ms=pct(0.50),
        latency_p95_ms=pct(0.95),
        latency_p99_ms=pct(0.99),
        prediction_counts=dict(METRICS["predictions"]),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["inference"],
          responses={415: {"description": "Unsupported media type"},
                     413: {"description": "File too large"},
                     422: {"description": "Undecodable image"}})
async def predict(file: UploadFile = File(..., description="An image of one shape")):
    """Classify a single uploaded image."""
    bundle = get_bundle()
    t0 = time.perf_counter()
    METRICS["requests"] += 1

    raw = await read_validated(file)
    try:
        tensor = bundle.bytes_to_tensor(raw)
    except ValueError as e:
        METRICS["errors"] += 1
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail=str(e)) from e

    probs, infer_ms = bundle.predict(tensor)
    label, confidence = bundle.top_prediction(probs[0])

    METRICS["latencies"].append((time.perf_counter() - t0) * 1000)
    METRICS["predictions"][label] += 1

    return PredictionResponse(
        predicted_class=label,
        confidence=round(confidence, 6),
        probabilities=bundle.as_dict(probs[0]),
        model_version=bundle.version,
        inference_ms=round(infer_ms, 3),
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["inference"])
async def predict_batch(files: list[UploadFile] = File(...)):
    """Classify several images in a single forward pass.

    This is meaningfully faster per image than calling /predict in a loop: the
    per-request Python and framework overhead is amortised over the batch.
    """
    bundle = get_bundle()
    if not files:
        raise HTTPException(status_code=422, detail="No files supplied.")
    if len(files) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"{len(files)} files exceeds the batch limit of {MAX_BATCH_SIZE}.",
        )

    METRICS["requests"] += 1
    t0 = time.perf_counter()

    tensors, names = [], []
    for f in files:
        raw = await read_validated(f)
        try:
            tensors.append(bundle.bytes_to_tensor(raw))
        except ValueError as e:
            METRICS["errors"] += 1
            raise HTTPException(status_code=422,
                                detail=f"{f.filename}: {e}") from e
        names.append(f.filename or "unnamed")

    batch = torch.cat(tensors, dim=0)
    probs, infer_ms = bundle.predict(batch)

    items = []
    for name, row in zip(names, probs):
        label, conf = bundle.top_prediction(row)
        METRICS["predictions"][label] += 1
        items.append(BatchItem(filename=name, predicted_class=label,
                               confidence=round(conf, 6),
                               probabilities=bundle.as_dict(row)))

    METRICS["latencies"].append((time.perf_counter() - t0) * 1000)
    return BatchPredictionResponse(
        predictions=items,
        count=len(items),
        model_version=bundle.version,
        inference_ms=round(infer_ms, 3),
        ms_per_image=round(infer_ms / len(items), 3),
    )


@app.get("/", tags=["ops"])
def root():
    return JSONResponse({
        "service": "Shape Classifier API",
        "docs": "/docs",
        "endpoints": ["/health", "/metrics", "/predict", "/predict/batch"],
    })
