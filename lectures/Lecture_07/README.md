# Lecture 07 — Data Augmentation, Training Recipes and Experiment Tracking

*The unglamorous work that produces most of the accuracy.*

Part II - Convolutional Vision · DL-601 Deep Learning

## Contents of this folder

| Path | What it is |
|---|---|
| `slides/Lecture_07_Data_Augmentation_Training_Recipes_and_Experiment_Tracking.pptx` | Lecture deck, ready to present |
| `notebooks/L07_Data_Augmentation_Training_Recipes_and_Exper.ipynb` | Hands-on lab, runs end to end on CPU |
| `tasks/Lecture_07_Tasks.md` | 6 core + 2 stretch tasks for students |
| `data/README.md` | Which datasets this lecture uses and how to load them |

## Learning objectives

- Design an augmentation policy whose transformations preserve the label.
- Implement mixup and cutmix and explain what they regularise.
- Run a reproducible experiment: fixed seeds, logged configuration, recorded artefacts.
- Choose between grid, random and successive-halving hyper-parameter search.

## Lecture outline

1. **Augmentation as a prior** — Augmentation encodes the invariances you know the task has.
2. **The standard geometric and photometric toolkit** — Random crop with padding, horizontal flip, small rotation, scale and translate.
3. **Mixup and cutmix** — Mixup: x = lam*x_i + (1-lam)*x_j with the labels mixed identically, lam ~ Beta(a, a).
4. **Test-time augmentation** — Average predictions over several augmented copies of the test image.
5. **Reproducibility in practice** — Seed Python, NumPy and PyTorch; set deterministic algorithms when you need bitwise repeatability.
6. **Hyper-parameter search** — Grid search wastes evaluations on parameters that do not matter.

## The lab

Build an augmentation pipeline from tensor operations, measure each transformation's effect in a controlled ablation, implement mixup and cutmix with soft-label loss, then build a small experiment tracker (JSON log + results table) and run a random search over the training recipe.

**Data:** `shapes_32.npz`

## Teaching notes

- Suggested timing: 2 hours lecture (sections 1–6), 1 hour supervised lab.
- The notebook is written to be run live; each section maps to a slide section.
- Cells that take longer than ~60 s on a laptop are marked in the notebook.
- Every figure in the notebook can be dropped straight into the deck if you want to extend it.

## Reading

- Zhang et al. (2018), *mixup: Beyond Empirical Risk Minimization*.
- Yun et al. (2019), *CutMix*.
- Bergstra & Bengio (2012), *Random Search for Hyper-Parameter Optimization*.
- Shorten & Khoshgoftaar (2019), *A Survey on Image Data Augmentation*.
