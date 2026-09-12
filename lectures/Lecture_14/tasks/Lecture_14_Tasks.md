# Lecture 14 — Tasks
## Vision Transformers and Self-Supervised Learning

> Images as sequences of patches, and learning without labels.

**Course:** DL-601 Deep Learning · **Part:** Part IV - Attention, Generation and Production  
**Weight:** 1.5% of the final grade · **Due:** before Lecture 15

---

## What you should be able to do after this

- Explain the ViT patch-embedding pipeline and where its inductive bias differs from a CNN's.
- Implement a Vision Transformer from scratch and train it on the shapes dataset.
- Describe contrastive (SimCLR, MoCo), distillation (DINO) and masked (MAE) self-supervision.
- Run a linear probe to quantify the quality of a learned representation.

## Before you start

```bash
# from the repository root
pip install -r requirements.txt
python tools/build_data.py          # only needed once
jupyter lab lectures/Lecture_14/notebooks/L14_Vision_Transformers_and_SelfSupervised_Learn.ipynb
```

**Data used:** `shapes_32.npz, shapes_pairs.npz`  
**Helpers:** `from dlcourse import load_shapes, train, evaluate, show_grid, plot_history`

Work through the notebook first — it builds the ideas these tasks assume. Every task below is marked `TODO` in the notebook at the point where it belongs.

---

## Core tasks (6 required)

All core tasks must be attempted. Each is worth an equal share of the task mark.

### Task 1 — Patch embedding

Implement PatchEmbed with a strided convolution. For a 32x32 input with patch size 4, verify the output is (B, 64, embed_dim) and explain the 64.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 2 — Vision Transformer

Assemble a ViT with the [CLS] token, learned positional embeddings and the encoder blocks from Lecture 13. Train it on shapes and report accuracy and parameter count.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 3 — ViT versus CNN under data scarcity

Train both at matched parameter counts on 500, 2000 and 6000 training examples. Plot accuracy against training-set size and explain the crossover in terms of inductive bias.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 4 — Attention distance

For each ViT layer, compute the mean attention distance in pixels. Plot it against depth and compare the early layers to a CNN's receptive field.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 5 — SimCLR augmentations and NT-Xent

Implement the two-view augmentation pipeline and the NT-Xent loss with temperature 0.5. Verify the loss on a hand-constructed batch where you know the right answer.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 6 — Pre-train and probe

Pre-train the encoder on shapes_pairs.npz for 20 epochs using no labels at all. Then train a linear probe on 100 labelled examples. Compare against supervised training on the same 100 examples.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

---

## Stretch tasks (2, optional)

Not marked, but these are where the subject gets interesting. Attempt at least one over the semester if you are aiming for an A.

### Stretch 1 — Augmentation ablation for SimCLR

Remove colour jitter, then remove random cropping. Report linear-probe accuracy for each and explain why cropping matters most.

### Stretch 2 — Masked autoencoder

Implement a minimal MAE: mask 75% of patches, reconstruct with a light decoder. Visualise reconstructions and report linear-probe accuracy against your SimCLR result.

---

## Submission checklist

- [ ] Notebook runs top to bottom from a restarted kernel with no errors.
- [ ] `set_seed(0)` is called before anything random happens.
- [ ] Every core task is answered, in order, under its own heading.
- [ ] Every plot has axis labels and a title.
- [ ] Written answers are in markdown cells, not in code comments.
- [ ] Results added to your running `results.md` table (carried across every lecture).
- [ ] Any AI-assistant use is disclosed in a short note at the end of the notebook.

Submit as: `LASTNAME_FIRSTNAME_L14.ipynb`

## How this is marked

| Criterion | Weight | What full marks looks like |
|---|---|---|
| Correctness | 40% | Every core task produces the required output, and the numbers are right. |
| Code quality | 20% | Readable, functions have docstrings, no copy-pasted blocks, no dead code. |
| Analysis | 25% | Written answers explain *why*, not just *what*. Claims are backed by a number or a plot. |
| Reproducibility | 15% | Runs top to bottom from a clean kernel with the seed fixed. Same numbers each time. |

## Reading

- Dosovitskiy et al. (2021), *An Image Is Worth 16x16 Words* (ViT).
- Chen et al. (2020), *A Simple Framework for Contrastive Learning* (SimCLR).
- He et al. (2022), *Masked Autoencoders Are Scalable Vision Learners*.
- Caron et al. (2021), *Emerging Properties in Self-Supervised Vision Transformers* (DINO).

---

**Next lecture —** 15: Generative Models: Autoencoders, VAEs, GANs and Diffusion. Learning the distribution, not just the decision boundary.
