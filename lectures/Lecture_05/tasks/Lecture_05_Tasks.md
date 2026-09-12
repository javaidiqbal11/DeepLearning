# Lecture 05 — Tasks
## Convolutional Neural Networks

> The right inductive bias for images, derived rather than asserted.

**Course:** DL-601 Deep Learning · **Part:** Part II - Convolutional Vision  
**Weight:** 1.5% of the final grade · **Due:** before Lecture 06

---

## What you should be able to do after this

- Explain convolution as local connectivity plus weight sharing, and count the parameters it saves.
- Compute output shapes for any kernel, stride, padding and dilation without guessing.
- Implement 2D convolution and max pooling from scratch, then match PyTorch's result.
- Reason about receptive field growth and design a CNN whose receptive field covers the input.

## Before you start

```bash
# from the repository root
pip install -r requirements.txt
python tools/build_data.py          # only needed once
jupyter lab lectures/Lecture_05/notebooks/L05_Convolutional_Neural_Networks.ipynb
```

**Data used:** `shapes_32.npz`  
**Helpers:** `from dlcourse import load_shapes, train, evaluate, show_grid, plot_history`

Work through the notebook first — it builds the ideas these tasks assume. Every task below is marked `TODO` in the notebook at the point where it belongs.

---

## Core tasks (6 required)

All core tasks must be attempted. Each is worth an equal share of the task mark.

### Task 1 — Shape arithmetic without running code

For a 32x32x3 input, compute the output shape and parameter count for: Conv2d(3,16,3,pad=1); MaxPool2d(2); Conv2d(16,32,5,stride=2,pad=2); Conv2d(32,32,3,dilation=2,pad=2). Then verify each with a forward pass.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 2 — Convolution from scratch

Implement conv2d_naive(x, w, b, stride, padding) with explicit loops, and conv2d_im2col using matrix multiplication. Both must match F.conv2d to within 1e-5. Report the speed-up of im2col.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 3 — Max pooling forward and backward

Implement max pooling and its backward pass. Verify the gradient routes entirely to the argmax position.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 4 — Train a CNN on shapes

Build a three-block CNN (conv-ReLU-pool) with a dense head. Train to at least 97% test accuracy and add the result to your running results table beside the Lecture 1 baselines.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 5 — Parameter accounting

Compare your CNN against an MLP with the same test accuracy. Report parameter counts for both and explain the ratio in terms of weight sharing.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 6 — Visualise layer-1 filters

Plot all first-layer kernels as RGB images. Identify at least two that respond to oriented edges and show their activation maps on one test image.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

---

## Stretch tasks (2, optional)

Not marked, but these are where the subject gets interesting. Attempt at least one over the semester if you are aiming for an A.

### Stretch 1 — Receptive field calculator

Write receptive_field(layers) that returns the RF size at each layer for a list of (kernel, stride, dilation) tuples. Verify empirically by finding which input pixels affect one output unit.

### Stretch 2 — Translation equivariance test

Shift a test image by 1, 4 and 8 pixels. Measure how the CNN's and the MLP's predictions change, and quantify the difference.

---

## Submission checklist

- [ ] Notebook runs top to bottom from a restarted kernel with no errors.
- [ ] `set_seed(0)` is called before anything random happens.
- [ ] Every core task is answered, in order, under its own heading.
- [ ] Every plot has axis labels and a title.
- [ ] Written answers are in markdown cells, not in code comments.
- [ ] Results added to your running `results.md` table (carried across all 16 lectures).
- [ ] Any AI-assistant use is disclosed in a short note at the end of the notebook.

Submit as: `LASTNAME_FIRSTNAME_L05.ipynb`

## How this is marked

| Criterion | Weight | What full marks looks like |
|---|---|---|
| Correctness | 40% | Every core task produces the required output, and the numbers are right. |
| Code quality | 20% | Readable, functions have docstrings, no copy-pasted blocks, no dead code. |
| Analysis | 25% | Written answers explain *why*, not just *what*. Claims are backed by a number or a plot. |
| Reproducibility | 15% | Runs top to bottom from a clean kernel with the seed fixed. Same numbers each time. |

## Reading

- LeCun et al. (1998), *Gradient-Based Learning Applied to Document Recognition* (LeNet-5).
- Dumoulin & Visin (2016), *A Guide to Convolution Arithmetic for Deep Learning*.
- CS231n notes: Convolutional Neural Networks.

---

**Next lecture —** 06: Modern CNN Architectures and Transfer Learning. ResNet, and why fine-tuning beats training from scratch almost every time.
