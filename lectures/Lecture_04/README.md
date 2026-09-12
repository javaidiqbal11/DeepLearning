# Lecture 04 — Training Deep Networks: Optimisation and Regularisation

*Everything between 'it runs' and 'it works'.*

Part I - Foundations · DL-601 Deep Learning

## Contents of this folder

| Path | What it is |
|---|---|
| `slides/Lecture_04_Training_Deep_Networks_Optimisation_and_Regularisation.pptx` | Lecture deck, ready to present |
| `notebooks/L04_Training_Deep_Networks_Optimisation_and_Regu.ipynb` | Hands-on lab, runs end to end on CPU |
| `tasks/Lecture_04_Tasks.md` | 5 core + 2 stretch tasks for students |
| `data/README.md` | Which datasets this lecture uses and how to load them |

## Learning objectives

- Compare SGD, momentum, RMSProp and Adam, and state when each is the right default.
- Choose an initialisation scheme that keeps activation variance stable with depth.
- Apply batch normalisation, dropout and weight decay, and explain what each actually does.
- Diagnose underfitting and overfitting from learning curves and respond correctly.

## Lecture outline

1. **From gradient descent to Adam** — SGD: cheap, noisy, and the noise itself is a useful regulariser.
2. **Learning rate: the hyper-parameter that matters most** — Too high diverges; too low crawls or stalls in a bad region.
3. **Initialisation** — All zeros breaks symmetry permanently - every unit computes the same thing forever.
4. **Normalisation layers** — BatchNorm standardises each channel over the batch, then applies a learned scale and shift.
5. **Regularisation** — L2 / weight decay pulls weights toward zero and prefers smoother functions.
6. **Reading the learning curves** — High train loss and high val loss: underfitting - more capacity, longer training, higher LR.

## The lab

Run controlled experiments: an optimiser bake-off on identical seeds, a learning-rate range test, an initialisation comparison measured by activation variance per layer, and an ablation of BatchNorm, dropout and weight decay on a deliberately overfitting setup.

**Data:** `shapes_32.npz`

## Teaching notes

- Suggested timing: 2 hours lecture (sections 1–6), 1 hour supervised lab.
- The notebook is written to be run live; each section maps to a slide section.
- Cells that take longer than ~60 s on a laptop are marked in the notebook.
- Every figure in the notebook can be dropped straight into the deck if you want to extend it.

## Reading

- Kingma & Ba (2015), *Adam: A Method for Stochastic Optimization*.
- He et al. (2015), *Delving Deep into Rectifiers* (He initialisation).
- Ioffe & Szegedy (2015), *Batch Normalization*.
- Loshchilov & Hutter (2019), *Decoupled Weight Decay Regularization* (AdamW).
