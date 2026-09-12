# Lecture 16 — Multimodal Learning and Production ML Systems

*Joining vision to language, and shipping the result responsibly.*

Part IV - Attention, Generation and Production · DL-601 Deep Learning

## Contents of this folder

| Path | What it is |
|---|---|
| `slides/Lecture_16_Multimodal_Learning_and_Production_ML_Systems.pptx` | Lecture deck, ready to present |
| `notebooks/L16_Multimodal_Learning_and_Production_ML_System.ipynb` | Hands-on lab, runs end to end on CPU |
| `tasks/Lecture_16_Tasks.md` | 8 core + 2 stretch tasks for students |
| `data/README.md` | Which datasets this lecture uses and how to load them |
| `app/` | FastAPI service built during this lecture |

## Learning objectives

- Explain contrastive vision-language pre-training and how it enables zero-shot classification.
- Implement a small CLIP-style dual encoder and perform zero-shot and retrieval tasks.
- Design a production ML system: versioning, monitoring, drift detection and rollback.
- Audit a model for robustness, fairness and calibration before release.

## Lecture outline

1. **Multimodal learning** — Fusion strategies: early (concatenate inputs), late (combine predictions), joint embedding.
2. **What multimodal buys you** — Zero-shot transfer to classes never explicitly labelled during training.
3. **Production architecture** — Offline training pipeline versus the online serving path - different constraints, different code.
4. **Monitoring** — Operational: latency percentiles, throughput, error rate, saturation.
5. **Pre-release audit** — Robustness: corruption benchmarks, adversarial examples, out-of-distribution detection.
6. **Where the field is going** — Scaling laws, and the growing dominance of compute-optimal training.

## The lab

Build a CLIP-style dual encoder over the shapes images and their captions, train it with symmetric InfoNCE, run zero-shot classification from text prompts and bidirectional retrieval; then build the production layer: a versioned model registry, a FastAPI service with monitoring and drift detection endpoints, and a complete pre-release audit with a model card.

**Data:** `shapes_32.npz, captions.csv, shapes_pairs.npz`

## Teaching notes

- Suggested timing: 2 hours lecture (sections 1–6), 1 hour supervised lab.
- The notebook is written to be run live; each section maps to a slide section.
- Cells that take longer than ~60 s on a laptop are marked in the notebook.
- Every figure in the notebook can be dropped straight into the deck if you want to extend it.

## Reading

- Radford et al. (2021), *Learning Transferable Visual Models From Natural Language Supervision* (CLIP).
- Mitchell et al. (2019), *Model Cards for Model Reporting*.
- Sculley et al. (2015), *Hidden Technical Debt in Machine Learning Systems*.
- Hendrycks & Dietterich (2019), *Benchmarking Neural Network Robustness to Common Corruptions*.
