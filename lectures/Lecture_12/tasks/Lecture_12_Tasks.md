# Lecture 12 — Tasks
## Sequence Models: RNNs, LSTMs and Image Captioning

> Memory, and the reason it eventually was not enough.

**Course:** DL-601 Deep Learning · **Part:** Part III - Vision Systems and Deployment  
**Weight:** 1.5% of the final grade · **Due:** before Lecture 13

---

## What you should be able to do after this

- Explain recurrence, parameter sharing across time, and backpropagation through time.
- Derive why vanilla RNN gradients vanish, and how gating in an LSTM or GRU addresses it.
- Build a character-level language model and a CNN-encoder / RNN-decoder captioning model.
- Articulate the sequential bottleneck that motivated the Transformer.

## Before you start

```bash
# from the repository root
pip install -r requirements.txt
python tools/build_data.py          # only needed once
jupyter lab lectures/Lecture_12/notebooks/L12_Sequence_Models_RNNs_LSTMs_and_Image_Caption.ipynb
```

**Data used:** `corpus.txt, sentiment.csv, captions.csv, shapes_32.npz`  
**Helpers:** `from dlcourse import load_shapes, train, evaluate, show_grid, plot_history`

Work through the notebook first — it builds the ideas these tasks assume. Every task below is marked `TODO` in the notebook at the point where it belongs.

---

## Core tasks (6 required)

All core tasks must be attempted. Each is worth an equal share of the task mark.

### Task 1 — RNN cell from scratch

Implement a vanilla RNN cell with explicit weight matrices. Verify one timestep against torch.nn.RNNCell with copied weights.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 2 — LSTM cell from scratch

Implement all four gates explicitly. Verify against torch.nn.LSTMCell with copied weights, and print the gate activations for one input to show what each gate is doing.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 3 — Character language model

Train a character-level LSTM on corpus.txt. Report perplexity and sample 300 characters at temperatures 0.5, 1.0 and 1.5. Comment on the difference.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 4 — Long-range copy task

Build a task requiring the model to reproduce a token seen T steps earlier. Plot accuracy against T for a vanilla RNN and an LSTM, for T in {5, 10, 25, 50, 100}.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 5 — Gradient flow comparison

Measure the gradient norm at timestep 0 for both architectures on a 100-step sequence. Report the ratio and connect it to the copy-task result.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

### Task 6 — Image captioning

Train a CNN encoder with an LSTM decoder on the shapes captions. Generate captions for eight test images with greedy decoding and report how many are factually correct.

**Deliverable:** working code in the notebook, plus the requested number, table or plot. Where the task asks you to explain something, write it in a markdown cell directly beneath the code — two or three sentences is the right length.

---

## Stretch tasks (2, optional)

Not marked, but these are where the subject gets interesting. Attempt at least one over the semester if you are aiming for an A.

### Stretch 1 — Beam search

Implement beam search with width 3 and length normalisation. Compare its captions against greedy decoding on the same images.

### Stretch 2 — Gradient clipping study

Train the vanilla RNN without clipping until the loss becomes NaN. Add clipping at norms 0.5, 1 and 5 and report which values keep training stable.

---

## Submission checklist

- [ ] Notebook runs top to bottom from a restarted kernel with no errors.
- [ ] `set_seed(0)` is called before anything random happens.
- [ ] Every core task is answered, in order, under its own heading.
- [ ] Every plot has axis labels and a title.
- [ ] Written answers are in markdown cells, not in code comments.
- [ ] Results added to your running `results.md` table (carried across every lecture).
- [ ] Any AI-assistant use is disclosed in a short note at the end of the notebook.

Submit as: `LASTNAME_FIRSTNAME_L12.ipynb`

## How this is marked

| Criterion | Weight | What full marks looks like |
|---|---|---|
| Correctness | 40% | Every core task produces the required output, and the numbers are right. |
| Code quality | 20% | Readable, functions have docstrings, no copy-pasted blocks, no dead code. |
| Analysis | 25% | Written answers explain *why*, not just *what*. Claims are backed by a number or a plot. |
| Reproducibility | 15% | Runs top to bottom from a clean kernel with the seed fixed. Same numbers each time. |

## Reading

- Hochreiter & Schmidhuber (1997), *Long Short-Term Memory*.
- Karpathy (2015), *The Unreasonable Effectiveness of Recurrent Neural Networks*.
- Vinyals et al. (2015), *Show and Tell: A Neural Image Caption Generator*.
- Pascanu et al. (2013), *On the Difficulty of Training Recurrent Neural Networks*.

---

**Next lecture —** 13: Attention and the Transformer. Scaled dot-product attention, built from nothing.
