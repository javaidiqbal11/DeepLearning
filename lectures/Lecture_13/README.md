# Lecture 13 — Attention and the Transformer

*Scaled dot-product attention, built from nothing.*

Part IV - Attention, Generation and Production · DL-601 Deep Learning

## Contents of this folder

| Path | What it is |
|---|---|
| `slides/Lecture_13_Attention_and_the_Transformer.pptx` | Lecture deck, ready to present |
| `notebooks/L13_Attention_and_the_Transformer.ipynb` | Hands-on lab, runs end to end on CPU |
| `tasks/Lecture_13_Tasks.md` | 7 core + 2 stretch tasks for students |
| `data/README.md` | Which datasets this lecture uses and how to load them |

## Learning objectives

- Derive scaled dot-product attention and explain every term, including the square-root scaling.
- Implement multi-head self-attention, positional encoding and a full encoder block from scratch.
- Distinguish encoder-only, decoder-only and encoder-decoder architectures and their uses.
- Analyse attention's quadratic cost and name the main mitigations.

## Lecture outline

1. **The idea** — Every position attends to every other position directly. Path length between any two tokens is 1.
2. **Multi-head attention** — Project into h subspaces, attend independently in each, concatenate, project out.
3. **Position** — Attention is permutation-equivariant - it has no notion of order at all.
4. **The Transformer block** — Sub-layer 1: multi-head self-attention. Sub-layer 2: position-wise feed-forward (usually 4x width).
5. **Three families** — Encoder-only (BERT): bidirectional context, for classification and retrieval.
6. **The cost** — Attention is O(n^2) in sequence length, in both time and memory.

## The lab

Implement scaled dot-product attention, multi-head attention, sinusoidal positional encoding and a full pre-norm encoder block from scratch, verifying each against torch.nn.MultiheadAttention; train a Transformer classifier on the sentiment corpus; visualise attention weights; and measure the quadratic scaling empirically.

**Data:** `sentiment.csv, corpus.txt`

## Teaching notes

- Suggested timing: 2 hours lecture (sections 1–6), 1 hour supervised lab.
- The notebook is written to be run live; each section maps to a slide section.
- Cells that take longer than ~60 s on a laptop are marked in the notebook.
- Every figure in the notebook can be dropped straight into the deck if you want to extend it.

## Reading

- Vaswani et al. (2017), *Attention Is All You Need*.
- Alammar, *The Illustrated Transformer*.
- Xiong et al. (2020), *On Layer Normalization in the Transformer Architecture*.
- Dao et al. (2022), *FlashAttention*.
