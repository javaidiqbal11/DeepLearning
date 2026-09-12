# Lecture 9 — Shape Classifier API

A complete FastAPI service for the model trained in Lectures 5–6. This is the reference
implementation students build toward.

## Run it

```bash
cd lectures/Lecture_09/app

python train_and_export.py      # ~2 min: trains, exports the artefact, writes samples
uvicorn main:app --reload       # http://127.0.0.1:8000/docs
```

```bash
curl http://127.0.0.1:8000/health
curl -F "file=@samples/circle.png" http://127.0.0.1:8000/predict
```

## Test and benchmark

```bash
pytest -v            # 18 tests: contract, error handling, golden outputs
python benchmark.py  # p50/p95/p99 latency, single vs batched
```

## Files

| File | What it does |
|---|---|
| `model.py` | Architecture, artefact loading, preprocessing, inference |
| `schemas.py` | Pydantic request/response models — these generate `/docs` |
| `main.py` | The FastAPI application |
| `train_and_export.py` | Offline: train and write a deployable artefact |
| `benchmark.py` | Latency measurement |
| `tests/test_api.py` | Endpoint and golden-output tests |
| `Dockerfile` | Multi-stage build, non-root user, healthcheck |

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness and readiness; reports the loaded model version |
| `GET` | `/metrics` | Latency percentiles and the prediction distribution |
| `POST` | `/predict` | One image → class probabilities |
| `POST` | `/predict/batch` | Up to 32 images in a single forward pass |

## Error contract

| Status | When |
|---|---|
| `415` | Content type is not an image |
| `413` | File over 5 MB, or batch over 32 files |
| `422` | Empty or undecodable image, or a missing file field |
| `503` | No model artefact loaded |

## Four details that matter more than they look

1. **The artefact carries its own preprocessing.** A `state_dict` alone is not
   deployable. If the normalisation statistics at serving differ from training, the
   service returns confident, wrong answers and nothing raises an exception. The
   notebook demonstrates this failure deliberately.

2. **The model loads once, in a lifespan handler.** Loading costs hundreds of
   milliseconds; doing it per request makes every request pay for it.

3. **`torch.set_num_threads(1)`.** Each request already runs in its own worker. Letting
   every worker spawn N threads oversubscribes the CPU and wrecks tail latency.

4. **`torch.inference_mode()` around the forward pass.** Without it PyTorch builds an
   autograd graph it will never use — wasted time and memory that accumulates under load.

## Measured on CPU

| Endpoint | p50 | p95 | Per image |
|---|---|---|---|
| `/predict` | ~33 ms | ~200 ms | 33 ms |
| `/predict/batch` (32) | ~253 ms | ~310 ms | **7.6 ms** |

Batching is ~4.4× faster per image: the per-request Python, HTTP parsing and framework
overhead is paid once per *request*, not once per image. The trade-off is queuing
latency — good for throughput, worse for any single request.

## Container

```bash
docker build -t shape-api .
docker run -p 8000:8000 shape-api
```
