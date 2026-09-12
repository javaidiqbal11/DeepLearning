# Lecture 05 — Convolutional Neural Networks

*The right inductive bias for images, derived rather than asserted.*

Part II - Convolutional Vision · DL-601 Deep Learning

## Contents of this folder

| Path | What it is |
|---|---|
| `slides/Lecture_05_Convolutional_Neural_Networks.pptx` | Lecture deck, ready to present |
| `notebooks/L05_Convolutional_Neural_Networks.ipynb` | Hands-on lab, runs end to end on CPU |
| `tasks/Lecture_05_Tasks.md` | 6 core + 2 stretch tasks for students |
| `data/README.md` | Which datasets this lecture uses and how to load them |

## Learning objectives

- Explain convolution as local connectivity plus weight sharing, and count the parameters it saves.
- Compute output shapes for any kernel, stride, padding and dilation without guessing.
- Implement 2D convolution and max pooling from scratch, then match PyTorch's result.
- Reason about receptive field growth and design a CNN whose receptive field covers the input.

## Lecture outline

1. **Why not just use an MLP** — A 224x224x3 image into one 1000-unit hidden layer is 150 million weights in a single layer.
2. **The two ideas** — Local connectivity: a unit sees a small patch, because pixels far apart are weakly related.
3. **Convolution arithmetic** — out = floor((in + 2*padding - dilation*(kernel - 1) - 1) / stride) + 1.
4. **Pooling and downsampling** — Max pooling gives small-translation invariance and reduces computation.
5. **Receptive field** — Stacking two 3x3 layers gives a 5x5 receptive field with fewer parameters than one 5x5 layer.
6. **What the filters learn** — Layer 1: oriented edges and colour blobs - remarkably close to Gabor filters, every time.

## The lab

Implement conv2d with im2col and max pooling from scratch and match torch.nn.functional to 1e-5; apply hand-built edge kernels to see what convolution does; train a CNN to ~99% and compare it against an MLP on accuracy, parameter count and — the decisive test — accuracy on translated images; then visualise the learned first-layer filters and measure the receptive field.

**Data:** `shapes_32.npz`

## Teaching notes

- Suggested timing: 2 hours lecture (sections 1–6), 1 hour supervised lab.
- The notebook is written to be run live; each section maps to a slide section.
- Cells that take longer than ~60 s on a laptop are marked in the notebook.
- Every figure in the notebook can be dropped straight into the deck if you want to extend it.

## Reading

- LeCun et al. (1998), *Gradient-Based Learning Applied to Document Recognition* (LeNet-5).
- Dumoulin & Visin (2016), *A Guide to Convolution Arithmetic for Deep Learning*.
- CS231n notes: Convolutional Neural Networks.
