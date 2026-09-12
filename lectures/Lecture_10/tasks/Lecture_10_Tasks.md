# Lecture 10 — Tasks
## Object Detection

> From 'what is in this image' to 'what, and exactly where'.

**Course:** DL-601 Deep Learning · **Part:** Part III - Vision Systems and Deployment  
**Weight:** 1.5% of the final grade · **Due:** before Lecture 11

---

## What you should be able to do after this

- Define the detection task and the role of IoU, non-maximum suppression and mean average precision.
- Contrast two-stage (R-CNN family) with one-stage (YOLO, SSD, RetinaNet) detectors.
- Implement a single-scale anchor-free detector with classification and box-regression heads.
- Compute mAP correctly and interpret a precision-recall curve.

## Before you start

```bash
# from the repository root
pip install -r requirements.txt
python tools/build_data.py          # only needed once
jupyter lab lectures/Lecture_10/notebooks/L10_Object_Detection.ipynb
```

**Data used:** `shapes_det.npz, shapes_det_annotations.json`  
**Helpers:** `from dlcourse import load_shapes, train, evaluate, show_grid, plot_history`

Work through the notebook first — it builds the ideas these tasks assume. Every task below is marked `TODO` in the notebook at the point where it belongs.

---

## Core tasks (6 required)

All core tasks must be attempted. Each is worth an equal share of the task mark.

### Task 1 — IoU, vectorised

Implement box_iou(boxes_a, boxes_b) returning the full pairwise matrix. Include tests for identical boxes (1.0), disjoint boxes (0.0), and one box contained in another.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 2 — Non-maximum suppression

Implement NMS from scratch and verify it against torchvision.ops.nms if available, otherwise against a hand-worked example of five overlapping boxes.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 3 — Build the detector

Implement a CNN backbone with two heads: a centre heatmap over a 24x24 grid and a 2-channel width/height regression. Train with focal loss on the heatmap and L1 on the sizes.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 4 — Decode predictions

Write decode(heatmap, sizes, threshold) that extracts peaks, converts grid coordinates to pixel boxes, and applies NMS. Visualise predictions against ground truth for eight validation scenes.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 5 — mean Average Precision

Implement AP at IoU 0.5 using the all-point interpolation rule, then mAP over the four classes. Plot the precision-recall curve for each class.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 6 — Confidence threshold sweep

Sweep the score threshold from 0.05 to 0.95. Plot precision and recall against threshold and pick an operating point, justifying it for a named use case.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

---

## Stretch tasks (2, optional)

Not marked, but these are where the subject gets interesting. Attempt at least one over the semester if you are aiming for an A.

### Stretch 1 — Focal loss ablation

Train with plain cross-entropy instead of focal loss. Report the mAP difference and explain it via the foreground/background ratio in your data.

### Stretch 2 — Multi-scale predictions

Add a second prediction head at a coarser stride. Report mAP separately for small and large objects.

---

## Submission checklist

- [ ] Notebook runs top to bottom from a restarted kernel with no errors.
- [ ] `set_seed(0)` is called before anything random happens.
- [ ] Every core task is answered, in order, under its own heading.
- [ ] Every plot has axis labels and a title.
- [ ] Written answers are in markdown cells, not in code comments.
- [ ] Results added to your running `results.md` table (carried across every lecture).
- [ ] Any AI-assistant use is disclosed in a short note at the end of the notebook.

Submit as: `LASTNAME_FIRSTNAME_L10.ipynb`

## How this is marked

| Criterion | Weight | What full marks looks like |
|---|---|---|
| Correctness | 40% | Every core task produces the required output, and the numbers are right. |
| Code quality | 20% | Readable, functions have docstrings, no copy-pasted blocks, no dead code. |
| Analysis | 25% | Written answers explain *why*, not just *what*. Claims are backed by a number or a plot. |
| Reproducibility | 15% | Runs top to bottom from a clean kernel with the seed fixed. Same numbers each time. |

## Reading

- Ren et al. (2015), *Faster R-CNN*.
- Redmon et al. (2016), *You Only Look Once*.
- Lin et al. (2017), *Focal Loss for Dense Object Detection*.
- Tian et al. (2019), *FCOS: Fully Convolutional One-Stage Object Detection*.

---

**Next lecture —** 11: Semantic Segmentation and Dense Prediction. One prediction per pixel, and the encoder-decoder that makes it possible.
