# Lecture 07 — Tasks
## Data Augmentation, Training Recipes and Experiment Tracking

> The unglamorous work that produces most of the accuracy.

**Course:** DL-601 Deep Learning · **Part:** Part II - Convolutional Vision  
**Weight:** 1.5% of the final grade · **Due:** before Lecture 08

---

## What you should be able to do after this

- Design an augmentation policy whose transformations preserve the label.
- Implement mixup and cutmix and explain what they regularise.
- Run a reproducible experiment: fixed seeds, logged configuration, recorded artefacts.
- Choose between grid, random and successive-halving hyper-parameter search.

## Before you start

```bash
# from the repository root
pip install -r requirements.txt
python tools/build_data.py          # only needed once
jupyter lab lectures/Lecture_07/notebooks/L07_Data_Augmentation_Training_Recipes_and_Exper.ipynb
```

**Data used:** `shapes_32.npz`  
**Helpers:** `from dlcourse import load_shapes, train, evaluate, show_grid, plot_history`

Work through the notebook first — it builds the ideas these tasks assume. Every task below is marked `TODO` in the notebook at the point where it belongs.

---

## Core tasks (6 required)

All core tasks must be attempted. Each is worth an equal share of the task mark.

### Task 1 — Augmentation from scratch

Implement random_crop_with_padding, random_horizontal_flip, random_rotation and colour_jitter as functions on CHW float tensors. Show a grid of 16 augmented copies of one image.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 2 — Label-preserving audit

For each of your transformations, state whether it preserves the label for the shapes dataset and justify it. Identify one transformation that would break the label and explain why.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 3 — Ablation study

Train with no augmentation, then adding one transformation at a time. Produce a table of validation accuracy and identify which transformation contributes most.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 4 — Mixup

Implement mixup with Beta(0.2, 0.2) and the corresponding soft-label cross-entropy. Train with it and report the change in validation accuracy and in expected calibration error.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 5 — Experiment tracker

Write a Tracker class that records config, per-epoch metrics, git commit and library versions to a JSON file per run, and a function that loads all runs into a sorted results DataFrame.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 6 — Random search

Run 12 random-search trials over learning rate (log-uniform 1e-4 to 1e-1), weight decay, dropout and augmentation strength. Report the best configuration and its validation accuracy.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

---

## Stretch tasks (2, optional)

Not marked, but these are where the subject gets interesting. Attempt at least one over the semester if you are aiming for an A.

### Stretch 1 — Cutmix

Implement cutmix, compare it against mixup at equal epoch budget, and show example mixed images with their soft labels.

### Stretch 2 — Test-time augmentation

Average predictions over 8 augmented copies at test time. Report the accuracy gain and the added latency per image in milliseconds.

---

## Submission checklist

- [ ] Notebook runs top to bottom from a restarted kernel with no errors.
- [ ] `set_seed(0)` is called before anything random happens.
- [ ] Every core task is answered, in order, under its own heading.
- [ ] Every plot has axis labels and a title.
- [ ] Written answers are in markdown cells, not in code comments.
- [ ] Results added to your running `results.md` table (carried across every lecture).
- [ ] Any AI-assistant use is disclosed in a short note at the end of the notebook.

Submit as: `LASTNAME_FIRSTNAME_L07.ipynb`

## How this is marked

| Criterion | Weight | What full marks looks like |
|---|---|---|
| Correctness | 40% | Every core task produces the required output, and the numbers are right. |
| Code quality | 20% | Readable, functions have docstrings, no copy-pasted blocks, no dead code. |
| Analysis | 25% | Written answers explain *why*, not just *what*. Claims are backed by a number or a plot. |
| Reproducibility | 15% | Runs top to bottom from a clean kernel with the seed fixed. Same numbers each time. |

## Reading

- Zhang et al. (2018), *mixup: Beyond Empirical Risk Minimization*.
- Yun et al. (2019), *CutMix*.
- Bergstra & Bengio (2012), *Random Search for Hyper-Parameter Optimization*.
- Shorten & Khoshgoftaar (2019), *A Survey on Image Data Augmentation*.

---

**Next lecture —** 08: Evaluation, Interpretability and Model Debugging. Accuracy is one number. It is rarely the one you need.
