# DL-601 — Deep Learning

**Foundations, Computer Vision and Production Systems**

A complete, ready-to-teach graduate module: 16 lectures, each with a slide deck, a
runnable notebook, its own data, and a marked task sheet.

📄 **[Deep_Learning_Course_Module.pdf](Deep_Learning_Course_Module.pdf)** — the full
course module document (28 pages: syllabus, assessment, per-lecture specifications).

---

## Quick start

```bash
# 1. dependencies
pip install -r requirements.txt

# 2. generate the datasets — once, ~30 seconds, fully offline
python tools/build_data.py

# 3. open the first lab
jupyter lab lectures/Lecture_01/notebooks/
```

That is the whole setup. **No downloads, no GPU, no API keys, no internet connection.**
Every dataset is generated procedurally by a seeded script, so the repository is
reproducible byte for byte.

---

## The 16 lectures

### Part I — Foundations
| # | Lecture | The idea |
|---|---|---|
| 01 | [Deep Learning Foundations and the Image Data Pipeline](lectures/Lecture_01/) | Images as tensors; baselines before models |
| 02 | [From Linear Models to Neural Networks](lectures/Lecture_02/) | Softmax, cross-entropy, and where linearity fails |
| 03 | [Backpropagation and Automatic Differentiation](lectures/Lecture_03/) | Build a working autodiff engine in 150 lines |
| 04 | [Training Deep Networks](lectures/Lecture_04/) | Optimisers, initialisation, normalisation, regularisation |

### Part II — Convolutional Vision
| # | Lecture | The idea |
|---|---|---|
| 05 | [Convolutional Neural Networks](lectures/Lecture_05/) | Weight sharing, receptive fields, translation equivariance |
| 06 | [Modern CNN Architectures and Transfer Learning](lectures/Lecture_06/) | ResNets, and why fine-tuning almost always wins |
| 07 | [Augmentation, Recipes and Experiment Tracking](lectures/Lecture_07/) | The unglamorous work that produces the accuracy |
| 08 | [Evaluation, Interpretability and Debugging](lectures/Lecture_08/) | Calibration, Grad-CAM, and catching a shortcut |

### Part III — Vision Systems and Deployment
| # | Lecture | The idea |
|---|---|---|
| 09 | [Serving Models with FastAPI](lectures/Lecture_09/) | A model nobody can call is not a deliverable |
| 10 | [Object Detection](lectures/Lecture_10/) | IoU, NMS, focal loss, mAP — all from scratch |
| 11 | [Semantic Segmentation](lectures/Lecture_11/) | U-Net, and what skip connections actually carry |
| 12 | [Sequence Models: RNNs, LSTMs, Captioning](lectures/Lecture_12/) | Memory, and why it was not enough |

### Part IV — Attention, Generation and Production
| # | Lecture | The idea |
|---|---|---|
| 13 | [Attention and the Transformer](lectures/Lecture_13/) | Scaled dot-product attention, built from nothing |
| 14 | [Vision Transformers and Self-Supervised Learning](lectures/Lecture_14/) | Patches as tokens; learning without labels |
| 15 | [Generative Models](lectures/Lecture_15/) | Autoencoders, VAEs, GANs, and a working DDPM |
| 16 | [Multimodal Learning and Production ML](lectures/Lecture_16/) | CLIP-style retrieval, drift detection, model cards |

---

## What is in each lecture folder

```
lectures/Lecture_NN/
├── README.md              objectives, outline, teaching notes, timing
├── slides/*.pptx          16:9 deck, ready to present
├── notebooks/*.ipynb      the lab — runs top to bottom on CPU
├── tasks/*_Tasks.md       core + stretch tasks, with the marking rubric
├── data/README.md         which datasets this lecture uses, and how to load them
└── app/                   FastAPI service (Lectures 9 and 16)
```

## Teaching approach

Every technique is **implemented from first principles before the library version is
used**. Over the semester students write, by hand:

- a reverse-mode autodiff engine, verified against PyTorch
- 2D convolution (naive and `im2col`) and max pooling with its backward pass
- residual blocks, and a reproduction of the degradation problem
- Grad-CAM via forward and backward hooks
- IoU, non-maximum suppression, focal loss and mean Average Precision
- U-Net, Dice loss and mean IoU
- RNN and LSTM cells, gate by gate
- scaled dot-product attention, multi-head attention and a Transformer block
- a Vision Transformer, and SimCLR with NT-Xent
- a VAE with the reparameterisation trick, a DCGAN, and a DDPM sampler
- a CLIP-style dual encoder with symmetric InfoNCE

They then **serve a model behind a tested HTTP API**, benchmark its tail latency, and
audit it for robustness, calibration and drift before "release".

## The running datasets

All generated offline by `tools/build_data.py`:

| Dataset | Shape | Used for |
|---|---|---|
| `shapes_32.npz` | 6000/1000/1000 × 32×32 RGB, 4 classes | Classification throughout |
| `shapes_imagefolder/` | 1000 PNG files in class folders | The file → tensor path |
| `digits_8x8.npz` | 1347/450 × 8×8, 10 classes | Real handwriting (from scikit-learn) |
| `shapes_det.npz` | 1500/300 × 96×96 scenes + boxes | Object detection |
| `shapes_seg.npz` | The same scenes + pixel masks | Segmentation |
| `shapes_pairs.npz` | 4000 unlabelled | Self-supervised pre-training |
| `sentiment.csv` | 3000 labelled sentences | Text classification |
| `corpus.txt` | ~250k characters | Character-level language modelling |
| `captions.csv` | 2000 image-caption pairs | Captioning and CLIP |

The shapes dataset is deliberately calibrated so that the teaching arc is real: chance
is 25%, a linear classifier reaches **63%**, and a small CNN reaches **99%**. That gap is
what the course explains.

## Rebuilding the course artefacts

Slides, notebooks, task sheets and the PDF are all generated from
[`tools/course_spec.py`](tools/course_spec.py), so the syllabus can never drift between
them.

```bash
python tools/build_data.py                 # datasets
python tools/build_slides.py               # 16 decks, 296 slides
python tools/lint_slides.py                # geometry and text-fit checks
python tools/build_notebooks.py            # 16 notebooks
python tools/build_notebooks.py --run      # ...and execute every one to prove it runs
python tools/build_tasks.py                # task sheets and READMEs
python tools/build_pdf.py                  # the course module PDF
```

`--run` executes every notebook end to end and fails the build if any cell raises. The
shipped notebooks are output-free so students run them fresh; executed copies land in
`build/executed/` for inspection.

## For instructors

- Each notebook maps section-for-section onto its slide deck — present the deck and run
  the matching cells live.
- Cells taking more than ~1 minute on a laptop are marked. Reduce the epoch count in
  class; students run the full version later.
- Suggested rhythm: 2 h lecture, 1 h supervised lab, 4–6 h independent work.
- Insist on the running `results.md` table from Lecture 1. It is the most useful thing
  students produce.

## Licence and attribution

Course materials prepared for graduate instruction. Curriculum design informed by
Stanford CS231n, NYU DS-GA 1008 (LeCun & Canziani), the deeplearning.ai Deep Learning
Specialization, fast.ai, and Goodfellow, Bengio & Courville's *Deep Learning*.
