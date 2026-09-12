# Lecture 16 — Tasks
## Multimodal Learning and Production ML Systems

> Joining vision to language, and shipping the result responsibly.

**Course:** DL-601 Deep Learning · **Part:** Part IV - Attention, Generation and Production  
**Weight:** 1.5% of the final grade · **Due:** with the capstone submission

---

## What you should be able to do after this

- Explain contrastive vision-language pre-training and how it enables zero-shot classification.
- Implement a small CLIP-style dual encoder and perform zero-shot and retrieval tasks.
- Design a production ML system: versioning, monitoring, drift detection and rollback.
- Audit a model for robustness, fairness and calibration before release.

## Before you start

```bash
# from the repository root
pip install -r requirements.txt
python tools/build_data.py          # only needed once
jupyter lab lectures/Lecture_16/notebooks/L16_Multimodal_Learning_and_Production_ML_System.ipynb
```

**Data used:** `shapes_32.npz, captions.csv, shapes_pairs.npz`  
**Helpers:** `from dlcourse import load_shapes, train, evaluate, show_grid, plot_history`

Work through the notebook first — it builds the ideas these tasks assume. Every task below is marked `TODO` in the notebook at the point where it belongs.

---

## Core tasks (8 required)

All core tasks must be attempted. Each is worth an equal share of the task mark.

### Task 1 — Dual encoder

Implement an image encoder (small CNN) and a text encoder (embedding + mean pooling or a small Transformer) projecting into a shared 64-dimensional L2-normalised space.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 2 — Symmetric contrastive loss

Implement InfoNCE in both directions with a learnable temperature. Verify that a perfectly aligned batch gives near-zero loss.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 3 — Zero-shot classification

Classify test images using only the text prompts 'a photo of a {class}'. Report accuracy and compare against the supervised Lecture 6 model.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 4 — Prompt sensitivity

Try five different prompt templates. Report the accuracy spread and comment on what that implies for deploying a zero-shot system.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 5 — Bidirectional retrieval

Implement text-to-image and image-to-text retrieval. Report Recall@1 and Recall@5 for both directions.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 6 — Model registry and serving

Build a registry that versions checkpoints with their metrics and config. Serve the latest approved version from FastAPI with GET /models and POST /models/{version}/promote.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 7 — Drift detection

Implement a /metrics endpoint tracking the prediction distribution and a population stability index against the training baseline. Feed it corrupted images and show the PSI alarm fire.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 8 — Model card

Write a complete model card: intended use, out-of-scope use, training data, evaluation results, subgroup performance, calibration, known limitations and the rollback procedure.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

---

## Stretch tasks (2, optional)

Not marked, but these are where the subject gets interesting. Attempt at least one over the semester if you are aiming for an A.

### Stretch 1 — Robustness benchmark

Evaluate under five corruptions (Gaussian noise, blur, brightness, contrast, rotation) at three severities. Produce a 5x3 accuracy table and identify the weakest axis.

### Stretch 2 — Quantisation

Apply dynamic quantisation to the served model. Report model size, p95 latency and accuracy before and after, and state whether you would ship it.

---

## Submission checklist

- [ ] Notebook runs top to bottom from a restarted kernel with no errors.
- [ ] `set_seed(0)` is called before anything random happens.
- [ ] Every core task is answered, in order, under its own heading.
- [ ] Every plot has axis labels and a title.
- [ ] Written answers are in markdown cells, not in code comments.
- [ ] Results added to your running `results.md` table (carried across every lecture).
- [ ] Any AI-assistant use is disclosed in a short note at the end of the notebook.

Submit as: `LASTNAME_FIRSTNAME_L16.ipynb`

## How this is marked

| Criterion | Weight | What full marks looks like |
|---|---|---|
| Correctness | 40% | Every core task produces the required output, and the numbers are right. |
| Code quality | 20% | Readable, functions have docstrings, no copy-pasted blocks, no dead code. |
| Analysis | 25% | Written answers explain *why*, not just *what*. Claims are backed by a number or a plot. |
| Reproducibility | 15% | Runs top to bottom from a clean kernel with the seed fixed. Same numbers each time. |

## Reading

- Radford et al. (2021), *Learning Transferable Visual Models From Natural Language Supervision* (CLIP).
- Mitchell et al. (2019), *Model Cards for Model Reporting*.
- Sculley et al. (2015), *Hidden Technical Debt in Machine Learning Systems*.
- Hendrycks & Dietterich (2019), *Benchmarking Neural Network Robustness to Common Corruptions*.
