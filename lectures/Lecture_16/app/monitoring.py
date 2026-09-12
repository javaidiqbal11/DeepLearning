"""Production monitoring: latency percentiles and label-free drift detection.

The central problem of production ML is that labels arrive late, or never, so
accuracy is usually not measurable live. What *is* measurable without labels is
whether the distribution of predictions has moved away from the baseline the
model was validated on. That is what this module tracks.
"""
from __future__ import annotations

import math
import time
from collections import Counter, deque

PSI_INVESTIGATE = 0.10
PSI_ALERT = 0.25


def population_stability_index(baseline: list[float], current: list[float],
                               eps: float = 1e-6) -> float:
    """PSI between two discrete distributions.

    PSI = sum_i (p_i - q_i) * ln(p_i / q_i)

    Interpretation, by long-standing convention in credit risk modelling where
    it originated: < 0.10 stable, 0.10-0.25 investigate, > 0.25 act.
    """
    if not baseline or not current or len(baseline) != len(current):
        return 0.0
    p = [v + eps for v in baseline]
    q = [v + eps for v in current]
    sp, sq = sum(p), sum(q)
    p = [v / sp for v in p]
    q = [v / sq for v in q]
    return sum((a - b) * math.log(a / b) for a, b in zip(p, q))


def psi_status(psi: float) -> str:
    if psi < PSI_INVESTIGATE:
        return "stable"
    if psi < PSI_ALERT:
        return "investigate"
    return "alert"


class Monitor:
    """Rolling operational and drift metrics for one served model."""

    def __init__(self, classes: list[str], baseline: list[float] | None = None,
                 window: int = 1000):
        self.classes = classes
        self.baseline = baseline or [1 / len(classes)] * len(classes)
        self.latencies: deque[float] = deque(maxlen=window)
        self.recent_predictions: deque[int] = deque(maxlen=window)
        self.prediction_counts: Counter[str] = Counter()
        self.confidences: deque[float] = deque(maxlen=window)
        self.requests = 0
        self.errors = 0
        self.started_at = time.time()

    # -- recording ---------------------------------------------------------
    def record(self, class_index: int, confidence: float, latency_ms: float) -> None:
        self.requests += 1
        self.recent_predictions.append(class_index)
        self.prediction_counts[self.classes[class_index]] += 1
        self.confidences.append(confidence)
        self.latencies.append(latency_ms)

    def record_error(self) -> None:
        self.requests += 1
        self.errors += 1

    # -- reporting ---------------------------------------------------------
    def percentile(self, p: float) -> float | None:
        if not self.latencies:
            return None
        s = sorted(self.latencies)
        return round(s[min(int(len(s) * p), len(s) - 1)], 3)

    def current_distribution(self) -> list[float]:
        if not self.recent_predictions:
            return [0.0] * len(self.classes)
        counts = [0] * len(self.classes)
        for c in self.recent_predictions:
            counts[c] += 1
        total = len(self.recent_predictions)
        return [c / total for c in counts]

    def drift(self) -> dict:
        current = self.current_distribution()
        psi = population_stability_index(self.baseline, current)
        return {
            "psi": round(psi, 5),
            "status": psi_status(psi),
            "baseline_distribution": {c: round(v, 4)
                                      for c, v in zip(self.classes, self.baseline)},
            "current_distribution": {c: round(v, 4)
                                     for c, v in zip(self.classes, current)},
            "samples_in_window": len(self.recent_predictions),
            "thresholds": {"investigate": PSI_INVESTIGATE, "alert": PSI_ALERT},
        }

    def snapshot(self) -> dict:
        mean_conf = (sum(self.confidences) / len(self.confidences)
                     if self.confidences else None)
        return {
            "uptime_seconds": round(time.time() - self.started_at, 2),
            "requests_total": self.requests,
            "errors_total": self.errors,
            "error_rate": round(self.errors / self.requests, 5) if self.requests else 0.0,
            "latency_p50_ms": self.percentile(0.50),
            "latency_p95_ms": self.percentile(0.95),
            "latency_p99_ms": self.percentile(0.99),
            "mean_confidence": round(mean_conf, 4) if mean_conf is not None else None,
            "prediction_counts": dict(self.prediction_counts),
            "drift": self.drift(),
        }
