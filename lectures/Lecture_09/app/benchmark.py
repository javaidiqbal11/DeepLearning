"""Latency benchmark for the classifier service.

    python benchmark.py

Uses TestClient rather than a real network socket, so it measures the service's
own cost without confounding it with network variance. Report p50, p95 and p99 —
the mean hides exactly the behaviour users complain about.
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

from fastapi.testclient import TestClient

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

from main import app  # noqa: E402

SAMPLES = APP_DIR / "samples"
N_REQUESTS = 200


def percentile(values, p):
    s = sorted(values)
    return s[min(int(len(s) * p), len(s) - 1)]


def summarise(name, latencies, per_image=None):
    print(f"\n{name}")
    print(f"  requests     : {len(latencies)}")
    print(f"  mean         : {statistics.mean(latencies):7.2f} ms")
    print(f"  p50          : {percentile(latencies, 0.50):7.2f} ms")
    print(f"  p95          : {percentile(latencies, 0.95):7.2f} ms")
    print(f"  p99          : {percentile(latencies, 0.99):7.2f} ms")
    print(f"  max          : {max(latencies):7.2f} ms")
    if per_image is not None:
        print(f"  per image    : {per_image:7.2f} ms")


def main():
    if not (APP_DIR / "artefacts" / "shape_classifier.pt").exists():
        sys.exit("No artefact found. Run `python train_and_export.py` first.")

    payload = (SAMPLES / "circle.png").read_bytes()

    with TestClient(app) as client:
        # Warm up: the first requests are always slower and would skew the tail.
        for _ in range(10):
            client.post("/predict", files={"file": ("c.png", payload, "image/png")})

        # --- single-image endpoint ---
        single = []
        for _ in range(N_REQUESTS):
            t0 = time.perf_counter()
            r = client.post("/predict", files={"file": ("c.png", payload, "image/png")})
            single.append((time.perf_counter() - t0) * 1000)
            assert r.status_code == 200
        summarise(f"POST /predict  ({N_REQUESTS} sequential requests)", single)

        # --- batch endpoint ---
        for batch_size in (4, 16, 32):
            files = [("files", (f"{i}.png", payload, "image/png")) for i in range(batch_size)]
            batched = []
            for _ in range(20):
                t0 = time.perf_counter()
                r = client.post("/predict/batch", files=files)
                batched.append((time.perf_counter() - t0) * 1000)
                assert r.status_code == 200, r.text
            summarise(f"POST /predict/batch  (batch size {batch_size}, 20 requests)",
                      batched, per_image=statistics.mean(batched) / batch_size)

        print("\nWhy batching wins: the per-request Python, HTTP parsing and framework")
        print("overhead is paid once per *request*, not once per image. The model's own")
        print("forward pass also vectorises better over a batch.")

        print("\n" + client.get("/metrics").text)


if __name__ == "__main__":
    main()
