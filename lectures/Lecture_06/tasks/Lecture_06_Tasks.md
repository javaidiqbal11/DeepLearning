# Lecture 06 — Tasks
## Modern CNN Architectures and Transfer Learning

> ResNet, and why fine-tuning beats training from scratch almost every time.

**Course:** DL-601 Deep Learning · **Part:** Part II - Convolutional Vision  
**Weight:** 1.5% of the final grade · **Due:** before Lecture 07

---

## What you should be able to do after this

- Trace the architectural lineage from LeNet through AlexNet, VGG and Inception to ResNet.
- Explain the degradation problem and how residual connections resolve it.
- Implement a residual block and a small ResNet from scratch.
- Choose correctly between feature extraction and fine-tuning given dataset size and domain distance.

## Before you start

```bash
# from the repository root
pip install -r requirements.txt
python tools/build_data.py          # only needed once
jupyter lab lectures/Lecture_06/notebooks/L06_Modern_CNN_Architectures_and_Transfer_Learni.ipynb
```

**Data used:** `shapes_32.npz, shapes_imagefolder/`  
**Helpers:** `from dlcourse import load_shapes, train, evaluate, show_grid, plot_history`

Work through the notebook first — it builds the ideas these tasks assume. Every task below is marked `TODO` in the notebook at the point where it belongs.

---

## Core tasks (5 required)

All core tasks must be attempted. Each is worth an equal share of the task mark.

### Task 1 — Residual block from scratch

Implement BasicBlock with two 3x3 convolutions, BatchNorm and a skip connection, including the 1x1 projection needed when channel count or stride changes. Verify output shapes for both cases.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 2 — Reproduce the degradation problem

Train a 20-layer and a 40-layer plain CNN. Show the deeper one has higher *training* loss. Add skip connections to both and show the ordering reverses.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 3 — Small ResNet on shapes

Assemble a ResNet-style network from your blocks. Reach at least 99% test accuracy and report parameter count and training time against the Lecture 5 CNN.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 4 — Feature extraction versus fine-tuning

Pre-train on circle/square only. Transfer to a triangle/star task with just 100 labelled examples, three ways: from scratch, frozen backbone, full fine-tuning at lr/10. Report all three accuracies.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 5 — How many layers to unfreeze

Sweep the number of unfrozen backbone blocks from 0 to all. Plot target-task accuracy against that number and state where the curve flattens.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

---

## Stretch tasks (2, optional)

Not marked, but these are where the subject gets interesting. Attempt at least one over the semester if you are aiming for an A.

### Stretch 1 — 1x1 bottleneck cost analysis

Compare parameters and FLOPs for a plain 3x3-3x3 block against a 1x1-3x3-1x1 bottleneck of equal input/output width. Report the ratio and verify with a forward-pass timing.

### Stretch 2 — Depthwise separable convolution

Implement it and substitute it into your ResNet. Report the accuracy lost and the parameters saved.

---

## Submission checklist

- [ ] Notebook runs top to bottom from a restarted kernel with no errors.
- [ ] `set_seed(0)` is called before anything random happens.
- [ ] Every core task is answered, in order, under its own heading.
- [ ] Every plot has axis labels and a title.
- [ ] Written answers are in markdown cells, not in code comments.
- [ ] Results added to your running `results.md` table (carried across all 16 lectures).
- [ ] Any AI-assistant use is disclosed in a short note at the end of the notebook.

Submit as: `LASTNAME_FIRSTNAME_L06.ipynb`

## How this is marked

| Criterion | Weight | What full marks looks like |
|---|---|---|
| Correctness | 40% | Every core task produces the required output, and the numbers are right. |
| Code quality | 20% | Readable, functions have docstrings, no copy-pasted blocks, no dead code. |
| Analysis | 25% | Written answers explain *why*, not just *what*. Claims are backed by a number or a plot. |
| Reproducibility | 15% | Runs top to bottom from a clean kernel with the seed fixed. Same numbers each time. |

## Reading

- He et al. (2016), *Deep Residual Learning for Image Recognition*.
- Simonyan & Zisserman (2015), *Very Deep Convolutional Networks* (VGG).
- Yosinski et al. (2014), *How Transferable Are Features in Deep Neural Networks?*
- Howard et al. (2017), *MobileNets*.

---

**Next lecture —** 07: Data Augmentation, Training Recipes and Experiment Tracking. The unglamorous work that produces most of the accuracy.
