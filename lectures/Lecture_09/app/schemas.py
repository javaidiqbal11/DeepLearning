"""Pydantic request and response models.

Declaring the response shape means FastAPI validates what you return, and the
OpenAPI docs at /docs are generated from these classes rather than written by
hand and left to rot.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    predicted_class: str = Field(..., description="Highest-probability class name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Probability of that class")
    probabilities: dict[str, float] = Field(..., description="Probability for every class")
    model_version: str = Field(..., description="Version of the model that produced this")
    inference_ms: float = Field(..., description="Model forward-pass time in milliseconds")

    model_config = {
        "protected_namespaces": (),        # allow the model_version field name
        "json_schema_extra": {
            "example": {
                "predicted_class": "circle",
                "confidence": 0.987,
                "probabilities": {"circle": 0.987, "square": 0.008,
                                  "triangle": 0.003, "star": 0.002},
                "model_version": "1.0.0",
                "inference_ms": 2.41,
            }
        },
    }


class BatchItem(BaseModel):
    filename: str
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]


class BatchPredictionResponse(BaseModel):
    predictions: list[BatchItem]
    count: int
    model_version: str
    inference_ms: float = Field(..., description="Total forward-pass time for the batch")
    ms_per_image: float

    model_config = {"protected_namespaces": ()}


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None
    classes: list[str] | None = None
    uptime_seconds: float

    model_config = {"protected_namespaces": ()}


class MetricsResponse(BaseModel):
    requests_total: int
    errors_total: int
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    latency_p99_ms: float | None
    prediction_counts: dict[str, int]

    model_config = {"protected_namespaces": ()}


class ErrorResponse(BaseModel):
    detail: str
