# Lecture 09 — Serving Deep Learning Models with FastAPI

*A model that nobody can call is a model that does not exist.*

Part III - Vision Systems and Deployment · DL-601 Deep Learning

## Contents of this folder

| Path | What it is |
|---|---|
| `slides/Lecture_09_Serving_Deep_Learning_Models_with_FastAPI.pptx` | Lecture deck, ready to present |
| `notebooks/L09_Serving_Deep_Learning_Models_with_FastAPI.ipynb` | Hands-on lab, runs end to end on CPU |
| `tasks/Lecture_09_Tasks.md` | 6 core + 2 stretch tasks for students |
| `data/README.md` | Which datasets this lecture uses and how to load them |
| `app/` | FastAPI service built during this lecture |

## Learning objectives

- Export a trained model and load it for inference with correct, reproducible preprocessing.
- Build a FastAPI service that accepts an image upload and returns calibrated class probabilities.
- Validate inputs with Pydantic and return useful, correct HTTP error codes.
- Measure latency and throughput, and improve both with batching and threading controls.

## Lecture outline

1. **Training code is not serving code** — Training optimises throughput on batches; serving optimises latency on single requests.
2. **Model export formats** — state_dict: the recommended PyTorch format, but it requires the class definition to load.
3. **FastAPI essentials** — Path operations, type hints and automatic OpenAPI documentation at /docs.
4. **Correctness at the edges** — Reject non-image uploads with 415, oversized files with 413, malformed input with 422.
5. **Performance** — Measure p50, p95 and p99 latency, not the mean. Tail latency is what users feel.
6. **Testing a model service** — TestClient gives you fast endpoint tests with no network.

## The lab

Take the Lecture 6 ResNet, export it with its preprocessing config, build a complete FastAPI service (upload, batch, health, metrics endpoints), write pytest tests with TestClient, and run a latency benchmark reporting p50/p95/p99 before and after batching.

**Data:** `shapes_32.npz, shapes_imagefolder/`

## Teaching notes

- Suggested timing: 2 hours lecture (sections 1–6), 1 hour supervised lab.
- The notebook is written to be run live; each section maps to a slide section.
- Cells that take longer than ~60 s on a laptop are marked in the notebook.
- Every figure in the notebook can be dropped straight into the deck if you want to extend it.

## Reading

- FastAPI documentation: Request Files, Dependencies, Testing, Lifespan Events.
- PyTorch documentation: Saving and Loading Models; TorchScript.
- Sculley et al. (2015), *Hidden Technical Debt in Machine Learning Systems*.
