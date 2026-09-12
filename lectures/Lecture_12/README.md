# Lecture 12 — Sequence Models: RNNs, LSTMs and Image Captioning

*Memory, and the reason it eventually was not enough.*

Part III - Vision Systems and Deployment · DL-601 Deep Learning

## Contents of this folder

| Path | What it is |
|---|---|
| `slides/Lecture_12_Sequence_Models_RNNs_LSTMs_and_Image_Captioning.pptx` | Lecture deck, ready to present |
| `notebooks/L12_Sequence_Models_RNNs_LSTMs_and_Image_Caption.ipynb` | Hands-on lab, runs end to end on CPU |
| `tasks/Lecture_12_Tasks.md` | 6 core + 2 stretch tasks for students |
| `data/README.md` | Which datasets this lecture uses and how to load them |

## Learning objectives

- Explain recurrence, parameter sharing across time, and backpropagation through time.
- Derive why vanilla RNN gradients vanish, and how gating in an LSTM or GRU addresses it.
- Build a character-level language model and a CNN-encoder / RNN-decoder captioning model.
- Articulate the sequential bottleneck that motivated the Transformer.

## Lecture outline

1. **Recurrence** — h_t = tanh(W_hh h_{t-1} + W_xh x_t + b) - the same weights at every timestep.
2. **Backpropagation through time** — Unroll the network over the sequence and apply standard backpropagation.
3. **LSTM** — A cell state runs through the sequence with only additive interactions - an uninterrupted gradient path.
4. **Text as input** — Tokenisation: character, word, or subword (BPE / WordPiece). Subword is the modern default.
5. **Image captioning** — CNN encoder produces a feature vector; RNN decoder generates tokens conditioned on it.
6. **The bottleneck** — Recurrence is inherently sequential - timestep t cannot start before t-1 finishes. No parallelism.

## The lab

Implement an RNN cell and an LSTM cell from scratch and verify against PyTorch; train a character-level language model on the corpus and sample from it; demonstrate the vanishing-gradient difference between RNN and LSTM on a long-range copy task; then build CNN-encoder / LSTM-decoder captioning on the shapes images with greedy and beam-search decoding.

**Data:** `corpus.txt, sentiment.csv, captions.csv, shapes_32.npz`

## Teaching notes

- Suggested timing: 2 hours lecture (sections 1–6), 1 hour supervised lab.
- The notebook is written to be run live; each section maps to a slide section.
- Cells that take longer than ~60 s on a laptop are marked in the notebook.
- Every figure in the notebook can be dropped straight into the deck if you want to extend it.

## Reading

- Hochreiter & Schmidhuber (1997), *Long Short-Term Memory*.
- Karpathy (2015), *The Unreasonable Effectiveness of Recurrent Neural Networks*.
- Vinyals et al. (2015), *Show and Tell: A Neural Image Caption Generator*.
- Pascanu et al. (2013), *On the Difficulty of Training Recurrent Neural Networks*.
