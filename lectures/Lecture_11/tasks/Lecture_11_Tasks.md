# Lecture 11 — Tasks
## Semantic Segmentation and Dense Prediction

> One prediction per pixel, and the encoder-decoder that makes it possible.

**Course:** DL-601 Deep Learning · **Part:** Part III - Vision Systems and Deployment  
**Weight:** 1.5% of the final grade · **Due:** before Lecture 12

---

## What you should be able to do after this

- Distinguish semantic, instance and panoptic segmentation and their evaluation metrics.
- Explain why an encoder-decoder with skip connections outperforms naive upsampling.
- Implement U-Net from scratch and train it on multi-object scenes.
- Compute mean IoU and Dice, and apply loss functions that handle class imbalance.

## Before you start

```bash
# from the repository root
pip install -r requirements.txt
python tools/build_data.py          # only needed once
jupyter lab lectures/Lecture_11/notebooks/L11_Semantic_Segmentation_and_Dense_Prediction.ipynb
```

**Data used:** `shapes_seg.npz`  
**Helpers:** `from dlcourse import load_shapes, train, evaluate, show_grid, plot_history`

Work through the notebook first — it builds the ideas these tasks assume. Every task below is marked `TODO` in the notebook at the point where it belongs.

---

## Core tasks (6 required)

All core tasks must be attempted. Each is worth an equal share of the task mark.

### Task 1 — U-Net implementation

Implement DoubleConv, Down, Up and OutConv blocks, and assemble a U-Net with depth 3. Verify that input and output spatial dimensions match for a 96x96 input.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 2 — Train and visualise

Train for 15 epochs and produce a figure with image, ground-truth mask and prediction for six validation scenes.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 3 — mIoU and Dice from scratch

Implement both metrics with a confusion-matrix accumulator. Report per-class IoU and the mean, and verify the background class is handled as you intend.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 4 — Loss comparison

Train three models with cross-entropy, Dice, and CE + Dice. Report mIoU for each and state which classes each loss favours.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 5 — Skip connections ablation

Remove the skip connections and retrain. Report the mIoU drop and show a side-by-side prediction demonstrating the loss of boundary detail.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 6 — Class imbalance

Report the pixel frequency of each class. Apply inverse-frequency class weights to cross-entropy and report the change in per-class IoU for the rarest class.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

---

## Stretch tasks (2, optional)

Not marked, but these are where the subject gets interesting. Attempt at least one over the semester if you are aiming for an A.

### Stretch 1 — Checkerboard artefacts

Replace bilinear upsampling with ConvTranspose2d(k=3, stride=2) and show the checkerboard artefacts in the output. Then fix them with k=4 and explain why.

### Stretch 2 — Boundary-aware evaluation

Implement boundary IoU (IoU restricted to a band around the object boundary). Report it alongside mIoU and explain what it reveals that mIoU does not.

---

## Submission checklist

- [ ] Notebook runs top to bottom from a restarted kernel with no errors.
- [ ] `set_seed(0)` is called before anything random happens.
- [ ] Every core task is answered, in order, under its own heading.
- [ ] Every plot has axis labels and a title.
- [ ] Written answers are in markdown cells, not in code comments.
- [ ] Results added to your running `results.md` table (carried across every lecture).
- [ ] Any AI-assistant use is disclosed in a short note at the end of the notebook.

Submit as: `LASTNAME_FIRSTNAME_L11.ipynb`

## How this is marked

| Criterion | Weight | What full marks looks like |
|---|---|---|
| Correctness | 40% | Every core task produces the required output, and the numbers are right. |
| Code quality | 20% | Readable, functions have docstrings, no copy-pasted blocks, no dead code. |
| Analysis | 25% | Written answers explain *why*, not just *what*. Claims are backed by a number or a plot. |
| Reproducibility | 15% | Runs top to bottom from a clean kernel with the seed fixed. Same numbers each time. |

## Reading

- Ronneberger et al. (2015), *U-Net: Convolutional Networks for Biomedical Image Segmentation*.
- Long et al. (2015), *Fully Convolutional Networks for Semantic Segmentation*.
- Chen et al. (2018), *DeepLabv3+*.
- Odena et al. (2016), *Deconvolution and Checkerboard Artifacts*.

---

**Next lecture —** 12: Sequence Models: RNNs, LSTMs and Image Captioning. Memory, and the reason it eventually was not enough.
