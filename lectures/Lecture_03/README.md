# Lecture 03 — Backpropagation and Automatic Differentiation

*The chain rule, organised as a graph - and built from scratch.*

Part I - Foundations · DL-601 Deep Learning

## Contents of this folder

| Path | What it is |
|---|---|
| `slides/Lecture_03_Backpropagation_and_Automatic_Differentiation.pptx` | Lecture deck, ready to present |
| `notebooks/L03_Backpropagation_and_Automatic_Differentiatio.ipynb` | Hands-on lab, runs end to end on CPU |
| `tasks/Lecture_03_Tasks.md` | 5 core + 2 stretch tasks for students |
| `data/README.md` | Which datasets this lecture uses and how to load them |

## Learning objectives

- Derive backpropagation as the reverse-mode application of the chain rule on a computational graph.
- Implement a working reverse-mode autodiff engine in under 150 lines of NumPy.
- Compute local gradients for the layers used throughout the course.
- Recognise vanishing and exploding gradients from the gradient-norm profile across layers.

## Lecture outline

1. **The computational graph** — Every model is a DAG of primitive operations; each node knows its local derivative.
2. **Forward mode versus reverse mode** — Forward mode costs one pass per input; reverse mode costs one pass per output.
3. **Local gradients you should know by heart** — Add: distributes the incoming gradient unchanged to both inputs.
4. **Broadcasting and the sum rule** — Where a tensor was broadcast in the forward pass, sum the gradient over the broadcast axes.
5. **When gradients go wrong** — Vanishing: repeated multiplication by factors below one drives early-layer gradients to zero.

## The lab

Build a Tensor class with reverse-mode autodiff from scratch (add, mul, matmul, relu, softmax cross-entropy), validate every gradient against PyTorch, train an MLP with it, then instrument a deep network to observe vanishing gradients directly.

**Data:** `digits_8x8.npz, shapes_32.npz`

## Teaching notes

- Suggested timing: 2 hours lecture (sections 1–5), 1 hour supervised lab.
- The notebook is written to be run live; each section maps to a slide section.
- Cells that take longer than ~60 s on a laptop are marked in the notebook.
- Every figure in the notebook can be dropped straight into the deck if you want to extend it.

## Reading

- Goodfellow et al., *Deep Learning*, Section 6.5.
- Baydin et al. (2018), *Automatic Differentiation in Machine Learning: a Survey*.
- Karpathy, *micrograd* - a 150-line autodiff engine worth reading line by line.
