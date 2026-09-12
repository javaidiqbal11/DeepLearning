"""Plotting helpers. Matplotlib only -- no seaborn dependency."""
from __future__ import annotations

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def _to_hwc(img) -> np.ndarray:
    """Accept CHW tensor, HWC array, or HW grayscale; return a plottable array."""
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = np.transpose(img, (1, 2, 0))
    if img.ndim == 3 and img.shape[-1] == 1:
        img = img[..., 0]
    if img.dtype != np.uint8:
        lo, hi = float(np.nanmin(img)), float(np.nanmax(img))
        if hi > 1.001 or lo < -0.001:            # un-normalised or standardised
            img = (img - lo) / max(hi - lo, 1e-8)
        img = np.clip(img, 0, 1)
    return img


def show_grid(images, labels=None, classes=None, n=16, ncols=8, title=None, cmap="gray"):
    """Plot the first ``n`` images in a grid with optional class captions."""
    n = min(n, len(images))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.5 * ncols, 1.7 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for i in range(len(axes)):
        axes[i].axis("off")
        if i >= n:
            continue
        axes[i].imshow(_to_hwc(images[i]), cmap=cmap)
        if labels is not None:
            lab = labels[i]
            lab = int(lab) if not isinstance(lab, str) else lab
            axes[i].set_title(classes[lab] if (classes and not isinstance(lab, str)) else lab,
                              fontsize=9)
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


def plot_history(hist, title="Training history"):
    """Loss and accuracy curves side by side -- the first thing to look at."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    ep = range(1, len(hist.train_loss) + 1)
    a1.plot(ep, hist.train_loss, label="train", marker="o", ms=3)
    if hist.val_loss:
        a1.plot(ep, hist.val_loss, label="val", marker="s", ms=3)
    a1.set_xlabel("epoch"); a1.set_ylabel("loss"); a1.set_title("Loss")
    a1.legend(); a1.grid(alpha=0.3)

    a2.plot(ep, hist.train_acc, label="train", marker="o", ms=3)
    if hist.val_acc:
        a2.plot(ep, hist.val_acc, label="val", marker="s", ms=3)
    a2.set_xlabel("epoch"); a2.set_ylabel("accuracy"); a2.set_title("Accuracy")
    a2.legend(); a2.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_confusion(y_true, y_pred, classes, normalize=True, title="Confusion matrix"):
    """Confusion matrix -- where accuracy alone hides the real failure mode."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    k = len(classes)
    cm = np.zeros((k, k), dtype=float)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    if normalize:
        cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(1.1 * k + 2.5, 1.1 * k + 2))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
    ax.set_xticks(range(k), classes, rotation=45, ha="right")
    ax.set_yticks(range(k), classes)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    thresh = cm.max() / 2
    for i in range(k):
        for j in range(k):
            ax.text(j, i, f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i, j])}",
                    ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


_BOX_COLOURS = ["#e74c3c", "#2980b9", "#27ae60", "#8e44ad", "#f39c12"]


def show_boxes(image, boxes, labels=None, classes=None, scores=None, ax=None, title=None):
    """Draw bounding boxes in [x1, y1, x2, y2] pixel coordinates."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(_to_hwc(image))
    ax.axis("off")
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [float(v) for v in box]
        lab = int(labels[i]) if labels is not None else 0
        colour = _BOX_COLOURS[lab % len(_BOX_COLOURS)]
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                       fill=False, edgecolor=colour, linewidth=2))
        if classes is not None:
            txt = classes[lab]
            if scores is not None:
                txt += f" {float(scores[i]):.2f}"
            ax.text(x1, max(y1 - 2, 0), txt, color="white", fontsize=8,
                    bbox=dict(facecolor=colour, edgecolor="none", pad=1))
    if title:
        ax.set_title(title, fontsize=10)
    return ax


def show_masks(images, masks, preds=None, n=4, classes=None):
    """Image / ground-truth mask / prediction, one row per sample."""
    n = min(n, len(images))
    ncols = 3 if preds is not None else 2
    fig, axes = plt.subplots(n, ncols, figsize=(3.0 * ncols, 3.0 * n))
    axes = np.atleast_2d(axes)
    titles = ["image", "ground truth", "prediction"]
    for i in range(n):
        panels = [_to_hwc(images[i]), masks[i]] + ([preds[i]] if preds is not None else [])
        for j, panel in enumerate(panels):
            p = panel.detach().cpu().numpy() if isinstance(panel, torch.Tensor) else np.asarray(panel)
            axes[i, j].imshow(_to_hwc(p) if j == 0 else p,
                              cmap=None if j == 0 else "tab10",
                              vmin=None if j == 0 else 0,
                              vmax=None if j == 0 else (len(classes) - 1 if classes else 4))
            axes[i, j].axis("off")
            if i == 0:
                axes[i, j].set_title(titles[j], fontsize=10)
    fig.tight_layout()
    return fig
