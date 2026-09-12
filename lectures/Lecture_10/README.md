# Lecture 10 — Object Detection

*From 'what is in this image' to 'what, and exactly where'.*

Part III - Vision Systems and Deployment · DL-601 Deep Learning

## Contents of this folder

| Path | What it is |
|---|---|
| `slides/Lecture_10_Object_Detection.pptx` | Lecture deck, ready to present |
| `notebooks/L10_Object_Detection.ipynb` | Hands-on lab, runs end to end on CPU |
| `tasks/Lecture_10_Tasks.md` | 6 core + 2 stretch tasks for students |
| `data/README.md` | Which datasets this lecture uses and how to load them |

## Learning objectives

- Define the detection task and the role of IoU, non-maximum suppression and mean average precision.
- Contrast two-stage (R-CNN family) with one-stage (YOLO, SSD, RetinaNet) detectors.
- Implement a single-scale anchor-free detector with classification and box-regression heads.
- Compute mAP correctly and interpret a precision-recall curve.

## Lecture outline

1. **The task and why it is harder** — Variable number of outputs per image - a classification head cannot express that.
2. **Intersection over Union** — IoU = area of overlap / area of union. The universal matching criterion.
3. **Two-stage detectors** — R-CNN: region proposals, then a CNN per region. Accurate and extremely slow.
4. **One-stage detectors** — YOLO: a single pass predicts boxes and classes on a grid. Fast enough for video.
5. **Non-maximum suppression** — Detectors emit many overlapping boxes for one object.
6. **Losses and evaluation** — Classification: cross-entropy or focal loss, over grid locations.

## The lab

Implement IoU and NMS from scratch with tests, build a single-scale anchor-free detector (a centre heatmap plus a size regression head) on the 96x96 multi-shape scenes, train it, decode its output, and implement mAP@0.5 to evaluate it properly.

**Data:** `shapes_det.npz, shapes_det_annotations.json`

## Teaching notes

- Suggested timing: 2 hours lecture (sections 1–6), 1 hour supervised lab.
- The notebook is written to be run live; each section maps to a slide section.
- Cells that take longer than ~60 s on a laptop are marked in the notebook.
- Every figure in the notebook can be dropped straight into the deck if you want to extend it.

## Reading

- Ren et al. (2015), *Faster R-CNN*.
- Redmon et al. (2016), *You Only Look Once*.
- Lin et al. (2017), *Focal Loss for Dense Object Detection*.
- Tian et al. (2019), *FCOS: Fully Convolutional One-Stage Object Detection*.
