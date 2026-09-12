# Lecture 06 — Modern CNN Architectures and Transfer Learning

*ResNet, and why fine-tuning beats training from scratch almost every time.*

Part II - Convolutional Vision · DL-601 Deep Learning

## Contents of this folder

| Path | What it is |
|---|---|
| `slides/Lecture_06_Modern_CNN_Architectures_and_Transfer_Learning.pptx` | Lecture deck, ready to present |
| `notebooks/L06_Modern_CNN_Architectures_and_Transfer_Learni.ipynb` | Hands-on lab, runs end to end on CPU |
| `tasks/Lecture_06_Tasks.md` | 5 core + 2 stretch tasks for students |
| `data/README.md` | Which datasets this lecture uses and how to load them |

## Learning objectives

- Trace the architectural lineage from LeNet through AlexNet, VGG and Inception to ResNet.
- Explain the degradation problem and how residual connections resolve it.
- Implement a residual block and a small ResNet from scratch.
- Choose correctly between feature extraction and fine-tuning given dataset size and domain distance.

## Lecture outline

1. **The architecture lineage** — LeNet-5 (1998): the template - conv, pool, conv, pool, dense.
2. **The degradation problem** — A 56-layer plain network had higher *training* error than a 20-layer one. Not overfitting - optimisation.
3. **Building blocks worth knowing** — 1x1 convolution: channel mixing and dimensionality reduction at negligible spatial cost.
4. **Transfer learning** — Early layers learn generic features (edges, textures) that transfer across almost any image domain.
5. **Choosing a strategy** — Small data, similar domain: freeze the backbone, train a linear head.

## The lab

Implement a residual block and a small ResNet, empirically reproduce the degradation problem with a 20- versus 40-layer plain network, then run the transfer-learning comparison: a source model pre-trained on one shape subset, transferred to a 100-example target task, versus training from scratch.

**Data:** `shapes_32.npz, shapes_imagefolder/`

## Teaching notes

- Suggested timing: 2 hours lecture (sections 1–5), 1 hour supervised lab.
- The notebook is written to be run live; each section maps to a slide section.
- Cells that take longer than ~60 s on a laptop are marked in the notebook.
- Every figure in the notebook can be dropped straight into the deck if you want to extend it.

## Reading

- He et al. (2016), *Deep Residual Learning for Image Recognition*.
- Simonyan & Zisserman (2015), *Very Deep Convolutional Networks* (VGG).
- Yosinski et al. (2014), *How Transferable Are Features in Deep Neural Networks?*
- Howard et al. (2017), *MobileNets*.
