# Lecture 08 — Evaluation, Interpretability and Model Debugging

*Accuracy is one number. It is rarely the one you need.*

Part II - Convolutional Vision · DL-601 Deep Learning

## Contents of this folder

| Path | What it is |
|---|---|
| `slides/Lecture_08_Evaluation_Interpretability_and_Model_Debugging.pptx` | Lecture deck, ready to present |
| `notebooks/L08_Evaluation_Interpretability_and_Model_Debugg.ipynb` | Hands-on lab, runs end to end on CPU |
| `tasks/Lecture_08_Tasks.md` | 6 core + 2 stretch tasks for students |
| `data/README.md` | Which datasets this lecture uses and how to load them |

## Learning objectives

- Select metrics appropriate to class imbalance and to the cost structure of the errors.
- Assess calibration and correct it with temperature scaling.
- Implement saliency maps and Grad-CAM and read them critically.
- Follow a systematic debugging protocol instead of changing hyper-parameters at random.

## Lecture outline

1. **Beyond accuracy** — With 99% negatives, a model predicting 'negative' always scores 99%. Accuracy is worthless there.
2. **The confusion matrix is the first diagnostic** — It converts one number into a map of which classes are being confused for which.
3. **Calibration** — A well-calibrated model that says 0.8 is right 80% of the time. Modern networks are badly over-confident.
4. **Interpretability methods** — Vanilla saliency: the gradient of the class score with respect to the input pixels. Noisy but instant.
5. **Shortcut learning** — Models exploit whatever correlates with the label, including artefacts you never intended.
6. **A debugging protocol** — 1. Overfit a single batch. If the loss will not reach zero, the bug is in the model or the loss.

## The lab

Build a full evaluation report for the Lecture 6 model: per-class metrics, confusion matrix, highest-confidence errors, reliability diagram with temperature scaling, saliency and Grad-CAM. Then diagnose a deliberately shortcut-corrupted dataset and prove the shortcut with Grad-CAM.

**Data:** `shapes_32.npz, shapes_det.npz`

## Teaching notes

- Suggested timing: 2 hours lecture (sections 1–6), 1 hour supervised lab.
- The notebook is written to be run live; each section maps to a slide section.
- Cells that take longer than ~60 s on a laptop are marked in the notebook.
- Every figure in the notebook can be dropped straight into the deck if you want to extend it.

## Reading

- Guo et al. (2017), *On Calibration of Modern Neural Networks*.
- Selvaraju et al. (2017), *Grad-CAM*.
- Adebayo et al. (2018), *Sanity Checks for Saliency Maps*.
- Geirhos et al. (2020), *Shortcut Learning in Deep Neural Networks*.
