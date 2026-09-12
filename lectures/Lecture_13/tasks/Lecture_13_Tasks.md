# Lecture 13 — Tasks
## Attention and the Transformer

> Scaled dot-product attention, built from nothing.

**Course:** DL-601 Deep Learning · **Part:** Part IV - Attention, Generation and Production  
**Weight:** 1.5% of the final grade · **Due:** before Lecture 14

---

## What you should be able to do after this

- Derive scaled dot-product attention and explain every term, including the square-root scaling.
- Implement multi-head self-attention, positional encoding and a full encoder block from scratch.
- Distinguish encoder-only, decoder-only and encoder-decoder architectures and their uses.
- Analyse attention's quadratic cost and name the main mitigations.

## Before you start

```bash
# from the repository root
pip install -r requirements.txt
python tools/build_data.py          # only needed once
jupyter lab lectures/Lecture_13/notebooks/L13_Attention_and_the_Transformer.ipynb
```

**Data used:** `sentiment.csv, corpus.txt`  
**Helpers:** `from dlcourse import load_shapes, train, evaluate, show_grid, plot_history`

Work through the notebook first — it builds the ideas these tasks assume. Every task below is marked `TODO` in the notebook at the point where it belongs.

---

## Core tasks (7 required)

All core tasks must be attempted. Each is worth an equal share of the task mark.

### Task 1 — Scaled dot-product attention

Implement attention(Q, K, V, mask=None) with correct masking. Verify against F.scaled_dot_product_attention to within 1e-5.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 2 — Why sqrt(d_k)

For d_k in {8, 64, 512}, sample random Q and K, and plot the distribution of the attention weights with and without the scaling. Show numerically that softmax saturates without it.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 3 — Multi-head attention

Implement MultiHeadAttention as a module. Verify against torch.nn.MultiheadAttention with copied weights, being careful about its packed in_proj_weight layout.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 4 — Positional encoding

Implement sinusoidal encoding and plot the resulting matrix as a heatmap. Then show that removing it entirely leaves a sentence-order task at chance accuracy.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 5 — Encoder block and classifier

Assemble a pre-norm encoder block and build a two-layer Transformer classifier. Train on sentiment.csv to at least 95% test accuracy.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 6 — Attention visualisation

Plot the attention matrix for each head on two example sentences. Describe in three sentences any pattern a head appears to have specialised in.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 7 — Causal masking

Implement a causal mask and prove by an ablation experiment that removing it lets a language model cheat - report the implausibly low loss it achieves.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

---

## Stretch tasks (2, optional)

Not marked, but these are where the subject gets interesting. Attempt at least one over the semester if you are aiming for an A.

### Stretch 1 — Quadratic scaling

Measure forward-pass time and peak memory for sequence lengths 64, 128, 256, 512 and 1024. Fit the exponent of the scaling curve and compare it to the theoretical 2.

### Stretch 2 — Pre-norm versus post-norm

Train a 6-layer Transformer in both configurations without warmup. Report which one diverges and explain why in terms of residual-stream magnitude.

---

## Submission checklist

- [ ] Notebook runs top to bottom from a restarted kernel with no errors.
- [ ] `set_seed(0)` is called before anything random happens.
- [ ] Every core task is answered, in order, under its own heading.
- [ ] Every plot has axis labels and a title.
- [ ] Written answers are in markdown cells, not in code comments.
- [ ] Results added to your running `results.md` table (carried across all 16 lectures).
- [ ] Any AI-assistant use is disclosed in a short note at the end of the notebook.

Submit as: `LASTNAME_FIRSTNAME_L13.ipynb`

## How this is marked

| Criterion | Weight | What full marks looks like |
|---|---|---|
| Correctness | 40% | Every core task produces the required output, and the numbers are right. |
| Code quality | 20% | Readable, functions have docstrings, no copy-pasted blocks, no dead code. |
| Analysis | 25% | Written answers explain *why*, not just *what*. Claims are backed by a number or a plot. |
| Reproducibility | 15% | Runs top to bottom from a clean kernel with the seed fixed. Same numbers each time. |

## Reading

- Vaswani et al. (2017), *Attention Is All You Need*.
- Alammar, *The Illustrated Transformer*.
- Xiong et al. (2020), *On Layer Normalization in the Transformer Architecture*.
- Dao et al. (2022), *FlashAttention*.

---

**Next lecture —** 14: Vision Transformers and Self-Supervised Learning. Images as sequences of patches, and learning without labels.
