# Lecture 04 — Tasks
## Training Deep Networks: Optimisation and Regularisation

> Everything between 'it runs' and 'it works'.

**Course:** DL-601 Deep Learning · **Part:** Part I - Foundations  
**Weight:** 1.5% of the final grade · **Due:** before Lecture 05

---

## What you should be able to do after this

- Compare SGD, momentum, RMSProp and Adam, and state when each is the right default.
- Choose an initialisation scheme that keeps activation variance stable with depth.
- Apply batch normalisation, dropout and weight decay, and explain what each actually does.
- Diagnose underfitting and overfitting from learning curves and respond correctly.

## Before you start

```bash
# from the repository root
pip install -r requirements.txt
python tools/build_data.py          # only needed once
jupyter lab lectures/Lecture_04/notebooks/L04_Training_Deep_Networks_Optimisation_and_Regu.ipynb
```

**Data used:** `shapes_32.npz`  
**Helpers:** `from dlcourse import load_shapes, train, evaluate, show_grid, plot_history`

Work through the notebook first — it builds the ideas these tasks assume. Every task below is marked `TODO` in the notebook at the point where it belongs.

---

## Core tasks (5 required)

All core tasks must be attempted. Each is worth an equal share of the task mark.

### Task 1 — Optimiser bake-off

Train the same MLP on shapes with SGD, SGD+momentum, RMSProp and Adam. Fix the seed and every other hyper-parameter. Plot four validation-loss curves on one axis and name the winner at 20 epochs.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 2 — Learning-rate range test

Sweep the learning rate from 1e-5 to 1e0 over one epoch, recording the loss after each batch. Plot loss against log learning rate and identify the largest rate that is still stable.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 3 — Initialisation and activation variance

For a 10-layer ReLU MLP initialised with zeros, N(0, 0.01), Xavier and He, plot the variance of each layer's activations. Explain why He keeps it flat.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 4 — Regularisation ablation

Take a model that overfits (train 99%, val 78%). Add weight decay, dropout, and both. Report a four-row table of train and validation accuracy and state which intervention paid off.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 5 — BatchNorm in train versus eval

Train a small CNN with BatchNorm. Evaluate the test set once in train() mode and once in eval() mode. Report both numbers and explain the difference in terms of batch statistics versus running statistics.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

---

## Stretch tasks (2, optional)

Not marked, but these are where the subject gets interesting. Attempt at least one over the semester if you are aiming for an A.

### Stretch 1 — Cosine schedule with warmup

Implement cosine annealing with linear warmup as a LambdaLR. Plot the rate over 50 epochs and compare final accuracy against a constant rate.

### Stretch 2 — Batch size and generalisation

Train at batch sizes 8, 64, 512 and 4096 with the learning rate scaled linearly. Report final validation accuracy and comment on the large-batch generalisation gap.

---

## Submission checklist

- [ ] Notebook runs top to bottom from a restarted kernel with no errors.
- [ ] `set_seed(0)` is called before anything random happens.
- [ ] Every core task is answered, in order, under its own heading.
- [ ] Every plot has axis labels and a title.
- [ ] Written answers are in markdown cells, not in code comments.
- [ ] Results added to your running `results.md` table (carried across every lecture).
- [ ] Any AI-assistant use is disclosed in a short note at the end of the notebook.

Submit as: `LASTNAME_FIRSTNAME_L04.ipynb`

## How this is marked

| Criterion | Weight | What full marks looks like |
|---|---|---|
| Correctness | 40% | Every core task produces the required output, and the numbers are right. |
| Code quality | 20% | Readable, functions have docstrings, no copy-pasted blocks, no dead code. |
| Analysis | 25% | Written answers explain *why*, not just *what*. Claims are backed by a number or a plot. |
| Reproducibility | 15% | Runs top to bottom from a clean kernel with the seed fixed. Same numbers each time. |

## Reading

- Kingma & Ba (2015), *Adam: A Method for Stochastic Optimization*.
- He et al. (2015), *Delving Deep into Rectifiers* (He initialisation).
- Ioffe & Szegedy (2015), *Batch Normalization*.
- Loshchilov & Hutter (2019), *Decoupled Weight Decay Regularization* (AdamW).

---

**Next lecture —** 05: Convolutional Neural Networks. The right inductive bias for images, derived rather than asserted.
