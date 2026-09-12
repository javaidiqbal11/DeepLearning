# Lecture 14 — Vision Transformers and Self-Supervised Learning

*Images as sequences of patches, and learning without labels.*

Part IV - Attention, Generation and Production · DL-601 Deep Learning

## Contents of this folder

| Path | What it is |
|---|---|
| `slides/Lecture_14_Vision_Transformers_and_SelfSupervised_Learning.pptx` | Lecture deck, ready to present |
| `notebooks/L14_Vision_Transformers_and_SelfSupervised_Learn.ipynb` | Hands-on lab, runs end to end on CPU |
| `tasks/Lecture_14_Tasks.md` | 6 core + 2 stretch tasks for students |
| `data/README.md` | Which datasets this lecture uses and how to load them |

## Learning objectives

- Explain the ViT patch-embedding pipeline and where its inductive bias differs from a CNN's.
- Implement a Vision Transformer from scratch and train it on the shapes dataset.
- Describe contrastive (SimCLR, MoCo), distillation (DINO) and masked (MAE) self-supervision.
- Run a linear probe to quantify the quality of a learned representation.

## Lecture outline

1. **Vision Transformer** — Split the image into fixed patches (16x16 in the original), flatten, and linearly project each.
2. **Inductive bias and the data requirement** — A CNN has locality and translation equivariance baked in. ViT has almost none.
3. **Why self-supervision** — Labels are expensive; unlabelled images are effectively free.
4. **Contrastive learning** — SimCLR: two augmented views of one image are positives; every other image in the batch is a negative.
5. **Beyond contrastive** — BYOL and DINO work without negatives, using a momentum teacher and stop-gradient to avoid collapse.
6. **Evaluating a representation** — Linear probe: freeze the encoder, train a linear classifier. The standard measure of representation quality.

## The lab

Implement patch embedding, the [CLS] token and a full ViT from scratch, train it on shapes and compare against the Lecture 6 CNN at matched parameter counts; then run SimCLR pre-training on the unlabelled pool and measure linear-probe accuracy against a supervised model trained on the same small label budget.

**Data:** `shapes_32.npz, shapes_pairs.npz`

## Teaching notes

- Suggested timing: 2 hours lecture (sections 1–6), 1 hour supervised lab.
- The notebook is written to be run live; each section maps to a slide section.
- Cells that take longer than ~60 s on a laptop are marked in the notebook.
- Every figure in the notebook can be dropped straight into the deck if you want to extend it.

## Reading

- Dosovitskiy et al. (2021), *An Image Is Worth 16x16 Words* (ViT).
- Chen et al. (2020), *A Simple Framework for Contrastive Learning* (SimCLR).
- He et al. (2022), *Masked Autoencoders Are Scalable Vision Learners*.
- Caron et al. (2021), *Emerging Properties in Self-Supervised Vision Transformers* (DINO).
