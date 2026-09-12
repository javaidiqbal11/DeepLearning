# Lecture 08 — Tasks
## Evaluation, Interpretability and Model Debugging

> Accuracy is one number. It is rarely the one you need.

**Course:** DL-601 Deep Learning · **Part:** Part II - Convolutional Vision  
**Weight:** 1.5% of the final grade · **Due:** before Lecture 09

---

## What you should be able to do after this

- Select metrics appropriate to class imbalance and to the cost structure of the errors.
- Assess calibration and correct it with temperature scaling.
- Implement saliency maps and Grad-CAM and read them critically.
- Follow a systematic debugging protocol instead of changing hyper-parameters at random.

## Before you start

```bash
# from the repository root
pip install -r requirements.txt
python tools/build_data.py          # only needed once
jupyter lab lectures/Lecture_08/notebooks/L08_Evaluation_Interpretability_and_Model_Debugg.ipynb
```

**Data used:** `shapes_32.npz, shapes_det.npz`  
**Helpers:** `from dlcourse import load_shapes, train, evaluate, show_grid, plot_history`

Work through the notebook first — it builds the ideas these tasks assume. Every task below is marked `TODO` in the notebook at the point where it belongs.

---

## Core tasks (6 required)

All core tasks must be attempted. Each is worth an equal share of the task mark.

### Task 1 — Full metric report

Compute per-class precision, recall and F1, plus macro and micro averages, without sklearn.metrics. Verify your numbers against sklearn afterwards.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 2 — Error gallery

Plot the 16 misclassified test images with the highest predicted confidence, captioned with true and predicted labels. Write three sentences on what they have in common.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 3 — Reliability diagram and ECE

Implement a 10-bin reliability diagram and expected calibration error. Report the ECE of your model.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 4 — Temperature scaling

Fit a single temperature on the validation set by minimising NLL. Report ECE before and after, and confirm accuracy is unchanged.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 5 — Grad-CAM

Implement Grad-CAM using forward and backward hooks on the last convolutional block. Produce overlays for one correct and one incorrect prediction per class.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 6 — Catch the shortcut

shortcut_data.py adds a small class-correlated marker to the corner of each training image. Train on it, observe the high validation accuracy, then use Grad-CAM to prove the model is reading the marker. Report accuracy on the clean test set.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

---

## Stretch tasks (2, optional)

Not marked, but these are where the subject gets interesting. Attempt at least one over the semester if you are aiming for an A.

### Stretch 1 — Occlusion sensitivity

Implement occlusion sensitivity with a sliding grey patch. Compare its map against Grad-CAM on the same image and comment on where they disagree.

### Stretch 2 — Saliency sanity check

Reproduce the Adebayo et al. model-randomisation test: compare saliency from a trained model against a randomly initialised one. Report whether your method passes.

---

## Submission checklist

- [ ] Notebook runs top to bottom from a restarted kernel with no errors.
- [ ] `set_seed(0)` is called before anything random happens.
- [ ] Every core task is answered, in order, under its own heading.
- [ ] Every plot has axis labels and a title.
- [ ] Written answers are in markdown cells, not in code comments.
- [ ] Results added to your running `results.md` table (carried across every lecture).
- [ ] Any AI-assistant use is disclosed in a short note at the end of the notebook.

Submit as: `LASTNAME_FIRSTNAME_L08.ipynb`

## How this is marked

| Criterion | Weight | What full marks looks like |
|---|---|---|
| Correctness | 40% | Every core task produces the required output, and the numbers are right. |
| Code quality | 20% | Readable, functions have docstrings, no copy-pasted blocks, no dead code. |
| Analysis | 25% | Written answers explain *why*, not just *what*. Claims are backed by a number or a plot. |
| Reproducibility | 15% | Runs top to bottom from a clean kernel with the seed fixed. Same numbers each time. |

## Reading

- Guo et al. (2017), *On Calibration of Modern Neural Networks*.
- Selvaraju et al. (2017), *Grad-CAM*.
- Adebayo et al. (2018), *Sanity Checks for Saliency Maps*.
- Geirhos et al. (2020), *Shortcut Learning in Deep Neural Networks*.

---

**Next lecture —** 09: Serving Deep Learning Models with FastAPI. A model that nobody can call is a model that does not exist.
