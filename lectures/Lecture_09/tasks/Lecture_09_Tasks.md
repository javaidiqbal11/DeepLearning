# Lecture 09 — Tasks
## Serving Deep Learning Models with FastAPI

> A model that nobody can call is a model that does not exist.

**Course:** DL-601 Deep Learning · **Part:** Part III - Vision Systems and Deployment  
**Weight:** 1.5% of the final grade · **Due:** before Lecture 10

---

## What you should be able to do after this

- Export a trained model and load it for inference with correct, reproducible preprocessing.
- Build a FastAPI service that accepts an image upload and returns calibrated class probabilities.
- Validate inputs with Pydantic and return useful, correct HTTP error codes.
- Measure latency and throughput, and improve both with batching and threading controls.

## Before you start

```bash
# from the repository root
pip install -r requirements.txt
python tools/build_data.py          # only needed once
jupyter lab lectures/Lecture_09/notebooks/L09_Serving_Deep_Learning_Models_with_FastAPI.ipynb
```

**Data used:** `shapes_32.npz, shapes_imagefolder/`  
**Helpers:** `from dlcourse import load_shapes, train, evaluate, show_grid, plot_history`

Work through the notebook first — it builds the ideas these tasks assume. Every task below is marked `TODO` in the notebook at the point where it belongs.

---

## Core tasks (6 required)

All core tasks must be attempted. Each is worth an equal share of the task mark.

### Task 1 — Export a deployable artefact

Save a checkpoint containing the state_dict, the class names, the normalisation statistics, the input size and a version string. Write load_model(path) that reconstructs the model and its preprocessing.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 2 — The predict endpoint

Implement POST /predict accepting an image upload and returning JSON with predicted class, all class probabilities, model version and inference time in milliseconds.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 3 — Input validation

Return 415 for a non-image content type, 413 for a file over 5 MB, and 422 for a corrupt image. Write a test for each case.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 4 — Batch endpoint

Implement POST /predict/batch accepting multiple files and running them as a single forward pass. Show it is faster per image than looping over /predict.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 5 — Test suite

Write at least six pytest tests with TestClient covering health, a successful prediction, each error case, and a golden-output test that pins one fixed image to one prediction.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 6 — Latency benchmark

Measure p50, p95 and p99 latency over 200 single-image requests. Repeat with the batch endpoint at batch size 16 and report per-image latency for both.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

---

## Stretch tasks (2, optional)

Not marked, but these are where the subject gets interesting. Attempt at least one over the semester if you are aiming for an A.

### Stretch 1 — TorchScript comparison

Export the model with torch.jit.script and serve it. Compare cold-start time and p95 latency against the state_dict version.

### Stretch 2 — Containerise

Write a Dockerfile using a slim Python base and a multi-stage build. Report the final image size and the container cold-start time.

---

## Submission checklist

- [ ] Notebook runs top to bottom from a restarted kernel with no errors.
- [ ] `set_seed(0)` is called before anything random happens.
- [ ] Every core task is answered, in order, under its own heading.
- [ ] Every plot has axis labels and a title.
- [ ] Written answers are in markdown cells, not in code comments.
- [ ] Results added to your running `results.md` table (carried across all 16 lectures).
- [ ] Any AI-assistant use is disclosed in a short note at the end of the notebook.

Submit as: `LASTNAME_FIRSTNAME_L09.ipynb`

## How this is marked

| Criterion | Weight | What full marks looks like |
|---|---|---|
| Correctness | 40% | Every core task produces the required output, and the numbers are right. |
| Code quality | 20% | Readable, functions have docstrings, no copy-pasted blocks, no dead code. |
| Analysis | 25% | Written answers explain *why*, not just *what*. Claims are backed by a number or a plot. |
| Reproducibility | 15% | Runs top to bottom from a clean kernel with the seed fixed. Same numbers each time. |

## Reading

- FastAPI documentation: Request Files, Dependencies, Testing, Lifespan Events.
- PyTorch documentation: Saving and Loading Models; TorchScript.
- Sculley et al. (2015), *Hidden Technical Debt in Machine Learning Systems*.

---

**Next lecture —** 10: Object Detection. From 'what is in this image' to 'what, and exactly where'.
