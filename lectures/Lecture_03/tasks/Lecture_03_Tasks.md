# Lecture 03 — Tasks
## Backpropagation and Automatic Differentiation

> The chain rule, organised as a graph - and built from scratch.

**Course:** DL-601 Deep Learning · **Part:** Part I - Foundations  
**Weight:** 1.5% of the final grade · **Due:** before Lecture 04

---

## What you should be able to do after this

- Derive backpropagation as the reverse-mode application of the chain rule on a computational graph.
- Implement a working reverse-mode autodiff engine in under 150 lines of NumPy.
- Compute local gradients for the layers used throughout the course.
- Recognise vanishing and exploding gradients from the gradient-norm profile across layers.

## Before you start

```bash
# from the repository root
pip install -r requirements.txt
python tools/build_data.py          # only needed once
jupyter lab lectures/Lecture_03/notebooks/L03_Backpropagation_and_Automatic_Differentiatio.ipynb
```

**Data used:** `digits_8x8.npz, shapes_32.npz`  
**Helpers:** `from dlcourse import load_shapes, train, evaluate, show_grid, plot_history`

Work through the notebook first — it builds the ideas these tasks assume. Every task below is marked `TODO` in the notebook at the point where it belongs.

---

## Core tasks (5 required)

All core tasks must be attempted. Each is worth an equal share of the task mark.

### Task 1 — Hand-derive a graph

For f(x, y, z) = (x + y) * max(z, 0) at x=2, y=-3, z=4, draw the graph, run the forward pass, and compute all three partial derivatives by hand. Show every intermediate value.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 2 — Finish the autodiff engine

The provided Tensor class has add, mul and matmul. Implement backward for relu, sum, mean and the log-sum-exp used by cross-entropy.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 3 — Verify against PyTorch

For each operation you implemented, build the same expression in PyTorch with requires_grad=True and assert the gradients agree to within 1e-6.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 4 — Train with your own engine

Train a two-layer MLP on digits_8x8 using only your engine - no torch.nn, no torch.optim. Reach at least 92% test accuracy.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 5 — Gradient-norm profile

Build a 15-layer sigmoid MLP. Plot the gradient norm of each layer on a log scale after one backward pass. Repeat with ReLU and with residual connections; put all three on one figure.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

---

## Stretch tasks (2, optional)

Not marked, but these are where the subject gets interesting. Attempt at least one over the semester if you are aiming for an A.

### Stretch 1 — Add Conv2d

Extend your engine with a 2D convolution and its backward pass. Verify against torch.nn.functional.conv2d on a random 4x3x8x8 input.

### Stretch 2 — Gradient checkpointing

Modify the engine to recompute activations instead of storing them. Measure peak memory and wall-clock time for both versions and report the trade-off.

---

## Submission checklist

- [ ] Notebook runs top to bottom from a restarted kernel with no errors.
- [ ] `set_seed(0)` is called before anything random happens.
- [ ] Every core task is answered, in order, under its own heading.
- [ ] Every plot has axis labels and a title.
- [ ] Written answers are in markdown cells, not in code comments.
- [ ] Results added to your running `results.md` table (carried across all 16 lectures).
- [ ] Any AI-assistant use is disclosed in a short note at the end of the notebook.

Submit as: `LASTNAME_FIRSTNAME_L03.ipynb`

## How this is marked

| Criterion | Weight | What full marks looks like |
|---|---|---|
| Correctness | 40% | Every core task produces the required output, and the numbers are right. |
| Code quality | 20% | Readable, functions have docstrings, no copy-pasted blocks, no dead code. |
| Analysis | 25% | Written answers explain *why*, not just *what*. Claims are backed by a number or a plot. |
| Reproducibility | 15% | Runs top to bottom from a clean kernel with the seed fixed. Same numbers each time. |

## Reading

- Goodfellow et al., *Deep Learning*, Section 6.5.
- Baydin et al. (2018), *Automatic Differentiation in Machine Learning: a Survey*.
- Karpathy, *micrograd* - a 150-line autodiff engine worth reading line by line.

---

**Next lecture —** 04: Training Deep Networks: Optimisation and Regularisation. Everything between 'it runs' and 'it works'.
