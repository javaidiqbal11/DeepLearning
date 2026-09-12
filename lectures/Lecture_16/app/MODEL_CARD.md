# Model Card — Shape Classifier

*Template and worked example for DL-601 Lecture 16. Students write their own for the
capstone. Numbers below come from `seed_registry.py` on CPU; regenerate them rather than
quoting them.*

## Overview

| | |
|---|---|
| **Model** | Shape Classifier |
| **Version** | 1.0.0 |
| **Type** | 4-class image classification (circle, square, triangle, star) |
| **Architecture** | Small ResNet — 3 stages, 2 basic blocks each, widths 16/32/64 |
| **Input** | 32×32 RGB, standardised with the training-split channel statistics |
| **Output** | Probability distribution over 4 classes |
| **Owner** | DL-601 teaching team |

## Intended use

Classifying images of a **single, centred geometric shape on a pale background**, within
the DL-601 teaching pipeline. It exists to demonstrate a complete training → serving →
monitoring loop.

## Out-of-scope use

- Photographs, or any natural image. The model has only ever seen synthetic renders.
- Images containing more than one shape — use the Lecture 10 detector.
- Shapes outside the four training classes. **There is no "unknown" output.** The model
  will confidently assign one of the four classes to anything it is given.
- Any decision that affects a person, in any way.

## Training data

- 6000 synthetic 32×32 RGB images from `tools/build_data.py` (seeded, reproducible).
- Balanced across the four classes (~1500 each).
- Position jitter ±5% of frame, scale 30–44% of frame, arbitrary rotation.
- **Colour is randomised and deliberately decorrelated from the label**, so the model
  cannot solve the task by reading hue.
- Gaussian pixel noise, σ ≈ 6/255.

Known gaps: no occlusion, no scale extremes, no cluttered backgrounds, no real-world
lighting or sensor noise.

## Evaluation

Held-out test split, 1000 images, never used for training or model selection.

| Metric | Value |
|---|---|
| Test accuracy | ~1.00 |
| Test loss | ~0.01 |

For reference, on the same test split: chance is 0.25, a logistic regression on raw
pixels reaches ~0.63.

### Robustness

Accuracy under corruption, measured in the Lecture 16 notebook:

| Corruption | Severity 0.2 | 0.5 | 1.0 |
|---|---|---|---|
| Gaussian noise | regenerate | regenerate | regenerate |
| Brightness shift | regenerate | regenerate | regenerate |
| Blur | regenerate | regenerate | regenerate |

Run notebook §6 to populate this table. Do not ship with it empty.

### Calibration

Expected Calibration Error measured on the in-distribution validation split. If ECE is
above ~0.05, apply temperature scaling (Lecture 8) before any downstream system is
allowed to act on the confidence value.

**Calibration was measured in-distribution only.** It says nothing about how the
confidence behaves on inputs unlike the training data.

## Known limitations

1. **Synthetic training data.** No claim of transfer to photographs is made or implied.
2. **No out-of-distribution detection.** Confidence stays high on inputs the model has
   never seen anything resembling. This is the most important limitation on this list.
3. **Fixed 32×32 input.** Larger uploads are downscaled, which destroys fine detail.
4. **Single-object assumption.** Multi-object scenes produce an arbitrary answer with no
   signal that the assumption was violated.

## Monitoring

| Signal | Source | Threshold |
|---|---|---|
| Prediction-distribution PSI | `GET /metrics` → `drift.psi` | investigate 0.10, alert 0.25 |
| p99 latency | `GET /metrics` | alert above 200 ms |
| Error rate | `GET /metrics` | alert above 1% |
| Mean confidence | `GET /metrics` | investigate on a sustained drop |

Accuracy is **not** monitored live, because labels are not available at serving time.
PSI is the proxy: it requires no labels, and it moves before accuracy does.

## Rollback

```bash
curl -X POST http://<host>/models/0.9.0/promote
```

Takes effect immediately; no restart required. The previous artefact is retained in the
registry and is never deleted on promotion. Verify with `GET /health`.

## Review triggers

Re-evaluate this model when any of these occur:

- PSI exceeds 0.25 for more than one hour
- The input distribution changes by design (new source, new capture pipeline)
- A new class needs to be supported — this requires retraining, not a config change
- Six months elapse with no review

## Changelog

| Version | Date | Change |
|---|---|---|
| 0.9.0 | — | First attempt; trained on 400 images. Archived. |
| 1.0.0 | — | Full 6000-image training set. Current production. |
