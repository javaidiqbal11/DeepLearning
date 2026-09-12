# Lecture 11 — Semantic Segmentation and Dense Prediction

*One prediction per pixel, and the encoder-decoder that makes it possible.*

Part III - Vision Systems and Deployment · DL-601 Deep Learning

## Contents of this folder

| Path | What it is |
|---|---|
| `slides/Lecture_11_Semantic_Segmentation_and_Dense_Prediction.pptx` | Lecture deck, ready to present |
| `notebooks/L11_Semantic_Segmentation_and_Dense_Prediction.ipynb` | Hands-on lab, runs end to end on CPU |
| `tasks/Lecture_11_Tasks.md` | 6 core + 2 stretch tasks for students |
| `data/README.md` | Which datasets this lecture uses and how to load them |

## Learning objectives

- Distinguish semantic, instance and panoptic segmentation and their evaluation metrics.
- Explain why an encoder-decoder with skip connections outperforms naive upsampling.
- Implement U-Net from scratch and train it on multi-object scenes.
- Compute mean IoU and Dice, and apply loss functions that handle class imbalance.

## Lecture outline

1. **Three segmentation tasks** — Semantic: one class label per pixel; two adjacent cars are one 'car' region.
2. **The resolution problem** — Classification backbones downsample aggressively - 32x is typical - to build semantic depth.
3. **Encoder-decoder with skip connections** — Encoder: downsample, growing the receptive field and the semantic content.
4. **Upsampling operators** — Nearest and bilinear interpolation: parameter-free, no checkerboard artefacts.
5. **Losses for dense prediction** — Pixel-wise cross-entropy is the default but is dominated by background pixels.
6. **Evaluation** — Per-class IoU = TP / (TP + FP + FN), then mean IoU over classes.

## The lab

Implement U-Net from scratch, train it on the 96x96 shape scenes for 5-class segmentation, compare cross-entropy against Dice against the combination, implement mean IoU and Dice metrics, and show the effect of removing the skip connections.

**Data:** `shapes_seg.npz`

## Teaching notes

- Suggested timing: 2 hours lecture (sections 1–6), 1 hour supervised lab.
- The notebook is written to be run live; each section maps to a slide section.
- Cells that take longer than ~60 s on a laptop are marked in the notebook.
- Every figure in the notebook can be dropped straight into the deck if you want to extend it.

## Reading

- Ronneberger et al. (2015), *U-Net: Convolutional Networks for Biomedical Image Segmentation*.
- Long et al. (2015), *Fully Convolutional Networks for Semantic Segmentation*.
- Chen et al. (2018), *DeepLabv3+*.
- Odena et al. (2016), *Deconvolution and Checkerboard Artifacts*.
