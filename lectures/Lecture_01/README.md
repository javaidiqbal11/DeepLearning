# Lecture 01 — Deep Learning Foundations and the Image Data Pipeline

*Why depth works, and how pixels become tensors.*

Part I - Foundations · DL-601 Deep Learning

## Contents of this folder

| Path | What it is |
|---|---|
| `slides/Lecture_01_Deep_Learning_Foundations_and_the_Image_Data_Pipeline.pptx` | Lecture deck, ready to present |
| `notebooks/L01_Deep_Learning_Foundations_and_the_Image_Data.ipynb` | Hands-on lab, runs end to end on CPU |
| `tasks/Lecture_01_Tasks.md` | 4 core + 2 stretch tasks for students |
| `data/README.md` | Which datasets this lecture uses and how to load them |

## Learning objectives

- Place deep learning relative to classical machine learning and explain what changed after 2012.
- Describe an image as a tensor and move fluently between NumPy, PIL and PyTorch layouts.
- Build a Dataset and DataLoader and reason about batching, shuffling and normalisation.
- Establish a baseline and a train/validation/test protocol before touching a model.

## Lecture outline

1. **What deep learning actually is** — Representation learning: features are learned, not hand-designed.
2. **When to use deep learning - and when not to** — Good fit: perceptual data (images, audio, text), abundant data, tolerance for opacity.
3. **Images as tensors** — A colour image is a rank-3 tensor: (height, width, channels).
4. **The data pipeline** — Dataset defines __len__ and __getitem__; DataLoader adds batching, shuffling and parallel workers.
5. **Splits, baselines and leakage** — Train fits parameters, validation selects hyper-parameters, test is touched exactly once.

## The lab

Load the shapes dataset three ways (npz arrays, a Dataset class, an on-disk image folder), visualise it, compute normalisation statistics correctly, and fit a majority-class and a linear baseline to fix the numbers the rest of the course must beat.

**Data:** `shapes_32.npz, shapes_imagefolder/`

## Teaching notes

- Suggested timing: 2 hours lecture (sections 1–5), 1 hour supervised lab.
- The notebook is written to be run live; each section maps to a slide section.
- Cells that take longer than ~60 s on a laptop are marked in the notebook.
- Every figure in the notebook can be dropped straight into the deck if you want to extend it.

## Reading

- Goodfellow, Bengio & Courville, *Deep Learning*, Chapter 1 and Section 5.1-5.3.
- Krizhevsky, Sutskever & Hinton (2012), *ImageNet Classification with Deep CNNs* (AlexNet).
- PyTorch docs: Datasets and DataLoaders tutorial.
