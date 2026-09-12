# Lecture 16 — Production ML Service

The Lecture 9 service grown up: a versioned model registry, live monitoring, and
label-free drift detection.

## Run it

```bash
cd lectures/Lecture_16/app

python seed_registry.py         # trains and registers v0.9.0 and v1.0.0
uvicorn main:app --reload       # http://127.0.0.1:8000/docs
pytest -v                       # registry, promotion, rollback, drift
```

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness; which version is serving |
| `GET` | `/metrics` | Latency percentiles, error rate, **drift report** |
| `GET` | `/models` | Every registered version and its stage |
| `POST` | `/models/{version}/promote` | Promote to production — also the rollback path |
| `POST` | `/predict` | One image → class probabilities |
| `GET` | `/model-card` | The model card, served alongside the model |

## The registry

Three rules, each enforced in code:

- **Versions are immutable.** Re-registering an existing version raises. Silent
  overwrite destroys the ability to answer "which model produced this prediction".
- **Exactly one version is in production.** Promotion archives the previous one.
- **Nothing is deleted on promotion.** Rollback is one call:
  `POST /models/0.9.0/promote`.

## Drift detection

You usually cannot measure accuracy in production, because labels arrive late or never.
What you *can* measure without labels is whether the distribution of predictions has
moved.

The Population Stability Index:

```
PSI = Σ (p_i − q_i) · ln(p_i / q_i)
```

| PSI | Status | Action |
|---|---|---|
| < 0.10 | stable | none |
| 0.10 – 0.25 | investigate | check inputs, look at recent samples |
| > 0.25 | alert | page someone; consider rollback |

`GET /metrics` returns the baseline distribution, the current windowed distribution, the
PSI and its status. The notebook shows PSI rising under Gaussian noise, brightness
shift, blur and a class-prior shift, and that it correlates strongly and negatively with
the accuracy you cannot see.

**Promotion resets the monitor.** A new model has a new baseline; carrying the old one
forward would raise a drift alert that is really just a deployment.

## Files

| File | What it does |
|---|---|
| `registry.py` | Versioned artefact store with staging → production → archived |
| `monitoring.py` | Latency percentiles, PSI, rolling prediction distribution |
| `main.py` | The FastAPI application |
| `seed_registry.py` | Trains and registers two versions so there is something to serve |
| `tests/test_production.py` | Registry, promotion, rollback and drift tests |

It reuses `model.py` from `lectures/Lecture_09/app/` rather than duplicating the
architecture — one definition, one place for it to be wrong.

## Before you would actually ship this

- [ ] Robustness benchmark across corruptions and severities (notebook §6)
- [ ] Calibration measured, and temperature scaling applied if needed (Lecture 8)
- [ ] Model card written: intended use, out-of-scope use, limitations, rollback
- [ ] Alert thresholds set from a real baseline, not from intuition
- [ ] Shadow deploy, then canary, then full rollout
- [ ] Rollback path tested — not assumed
