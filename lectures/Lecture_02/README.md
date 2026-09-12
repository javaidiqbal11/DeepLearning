# Lecture 02 — From Linear Models to Neural Networks

*The softmax classifier, and the exact point where linearity fails.*

Part I - Foundations · DL-601 Deep Learning

## Contents of this folder

| Path | What it is |
|---|---|
| `slides/Lecture_02_From_Linear_Models_to_Neural_Networks.pptx` | Lecture deck, ready to present |
| `notebooks/L02_From_Linear_Models_to_Neural_Networks.ipynb` | Hands-on lab, runs end to end on CPU |
| `tasks/Lecture_02_Tasks.md` | 5 core + 2 stretch tasks for students |
| `data/README.md` | Which datasets this lecture uses and how to load them |

## Learning objectives

- Derive the softmax classifier and the cross-entropy loss from maximum likelihood.
- Implement the forward pass and analytic gradient of a linear classifier in NumPy.
- Explain why stacking linear layers without a non-linearity gains nothing.
- Show empirically that a hidden layer plus ReLU solves a problem a linear model cannot.

## Lecture outline

1. **The linear score function** — s = Wx + b maps a flattened image to one score per class.
2. **Softmax and cross-entropy** — Softmax turns scores into a probability distribution: p_k = exp(s_k) / sum_j exp(s_j).
3. **Why a non-linearity is required** — W2(W1 x) = (W2 W1) x - composing linear maps yields another linear map.
4. **The universal approximation theorem, honestly** — One hidden layer of sufficient width can approximate any continuous function on a compact set.
5. **Loss surfaces and what optimisation faces** — Convex for a linear model with cross-entropy; non-convex the moment a hidden layer appears.

## The lab

Implement a softmax classifier in pure NumPy with an analytic gradient, verify it against a numerical gradient, train it on the digits data, then demonstrate the linear/non-linear gap on a two-moons problem and on the shapes dataset.

**Data:** `digits_8x8.npz, shapes_32.npz`

## Teaching notes

- Suggested timing: 2 hours lecture (sections 1–5), 1 hour supervised lab.
- The notebook is written to be run live; each section maps to a slide section.
- Cells that take longer than ~60 s on a laptop are marked in the notebook.
- Every figure in the notebook can be dropped straight into the deck if you want to extend it.

## Reading

- Goodfellow et al., *Deep Learning*, Chapter 6.
- CS231n course notes: Linear Classification and Optimization.
- Nair & Hinton (2010), *Rectified Linear Units Improve Restricted Boltzmann Machines*.
