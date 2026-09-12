# Lecture 02 — Tasks
## From Linear Models to Neural Networks

> The softmax classifier, and the exact point where linearity fails.

**Course:** DL-601 Deep Learning · **Part:** Part I - Foundations  
**Weight:** 1.5% of the final grade · **Due:** before Lecture 03

---

## What you should be able to do after this

- Derive the softmax classifier and the cross-entropy loss from maximum likelihood.
- Implement the forward pass and analytic gradient of a linear classifier in NumPy.
- Explain why stacking linear layers without a non-linearity gains nothing.
- Show empirically that a hidden layer plus ReLU solves a problem a linear model cannot.

## Before you start

```bash
# from the repository root
pip install -r requirements.txt
python tools/build_data.py          # only needed once
jupyter lab lectures/Lecture_02/notebooks/L02_From_Linear_Models_to_Neural_Networks.ipynb
```

**Data used:** `digits_8x8.npz, shapes_32.npz`  
**Helpers:** `from dlcourse import load_shapes, train, evaluate, show_grid, plot_history`

Work through the notebook first — it builds the ideas these tasks assume. Every task below is marked `TODO` in the notebook at the point where it belongs.

---

## Core tasks (5 required)

All core tasks must be attempted. Each is worth an equal share of the task mark.

### Task 1 — Numerically stable softmax

Implement softmax(scores) that handles a row containing 10000 without producing NaN. Show the naive version overflowing and yours not.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 2 — Analytic gradient plus gradient check

Implement the cross-entropy gradient for a linear classifier. Verify against a central-difference numerical gradient; relative error must be below 1e-6. Report the worst element.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 3 — Train the linear classifier

Train your NumPy softmax classifier on digits_8x8 with mini-batch gradient descent. Reach at least 90% test accuracy and plot the loss curve.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 4 — Visualise the templates

Reshape each row of the learned W back to 8x8 and plot all ten. Describe in two sentences what the template for the digit 0 has learned to detect.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 5 — One hidden layer changes everything

On sklearn's two-moons dataset, fit a linear classifier and an MLP with one hidden layer of 32 ReLU units. Plot both decision boundaries side by side and report both accuracies.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

---

## Stretch tasks (2, optional)

Not marked, but these are where the subject gets interesting. Attempt at least one over the semester if you are aiming for an A.

### Stretch 1 — Activation comparison

Train the same MLP on the shapes dataset with ReLU, tanh, sigmoid and LeakyReLU. Plot the four loss curves on one axis and explain the sigmoid result in terms of gradient saturation.

### Stretch 2 — Depth without non-linearity

Build a five-layer network with no activation functions. Show that its test accuracy matches a single linear layer, and confirm the product of its weight matrices is a rank-limited linear map.

---

## Submission checklist

- [ ] Notebook runs top to bottom from a restarted kernel with no errors.
- [ ] `set_seed(0)` is called before anything random happens.
- [ ] Every core task is answered, in order, under its own heading.
- [ ] Every plot has axis labels and a title.
- [ ] Written answers are in markdown cells, not in code comments.
- [ ] Results added to your running `results.md` table (carried across every lecture).
- [ ] Any AI-assistant use is disclosed in a short note at the end of the notebook.

Submit as: `LASTNAME_FIRSTNAME_L02.ipynb`

## How this is marked

| Criterion | Weight | What full marks looks like |
|---|---|---|
| Correctness | 40% | Every core task produces the required output, and the numbers are right. |
| Code quality | 20% | Readable, functions have docstrings, no copy-pasted blocks, no dead code. |
| Analysis | 25% | Written answers explain *why*, not just *what*. Claims are backed by a number or a plot. |
| Reproducibility | 15% | Runs top to bottom from a clean kernel with the seed fixed. Same numbers each time. |

## Reading

- Goodfellow et al., *Deep Learning*, Chapter 6.
- CS231n course notes: Linear Classification and Optimization.
- Nair & Hinton (2010), *Rectified Linear Units Improve Restricted Boltzmann Machines*.

---

**Next lecture —** 03: Backpropagation and Automatic Differentiation. The chain rule, organised as a graph - and built from scratch.
