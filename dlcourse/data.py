"""Dataset loading. Everything reads from the repo-level ``data/`` folder."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SHAPE_CLASSES = ["circle", "square", "triangle", "star"]

# ImageNet-style statistics computed on the shapes training split.
SHAPE_MEAN = (0.878, 0.874, 0.876)
SHAPE_STD = (0.168, 0.176, 0.180)


def _require(name: str) -> Path:
    p = DATA_DIR / name
    if not p.exists():
        raise FileNotFoundError(
            f"{p} is missing. Run `python tools/build_data.py` from the repo root."
        )
    return p


def to_nchw(x) -> torch.Tensor:
    """Return a float32 NCHW tensor in [0, 1], whatever layout you hand it.

    Accepts NHWC (how the generated datasets are stored), NHW grayscale, or
    data that is already NCHW. Detecting the layout rather than assuming it
    matters: silently permuting an NCHW batch produces a tensor of the wrong
    shape that only fails several layers into the model, if at all.
    """
    t = x if isinstance(x, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(x))

    if t.ndim == 3:                      # N, H, W  -> add the channel axis
        t = t.unsqueeze(1)
    elif t.ndim == 4:
        channels_last = t.shape[-1] in (1, 3) and t.shape[1] not in (1, 3)
        if channels_last:
            t = t.permute(0, 3, 1, 2)
        # otherwise it is already N, C, H, W and needs no reordering
    else:
        raise ValueError(f"expected a 3D or 4D batch, got shape {tuple(t.shape)}")

    t = t.float()
    if t.max() > 1.5:                    # uint8-valued data
        t = t / 255.0
    return t.contiguous()


class ArrayImageDataset(Dataset):
    """Wraps in-memory arrays, applying an optional transform per sample.

    Transforms receive and return a CHW float tensor, which keeps augmentation
    code in the notebooks free of PIL conversions.
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray | None = None, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        x = to_nchw(self.images[i: i + 1])[0]
        if self.transform is not None:
            x = self.transform(x)
        if self.labels is None:
            return x
        return x, int(self.labels[i])


def load_shapes(as_tensors: bool = True, normalize: bool = False):
    """The 4-class 32x32 shape dataset. Returns a dict of splits."""
    z = np.load(_require("shapes_32.npz"), allow_pickle=True)
    out = {"classes": [str(c) for c in z["classes"]]}
    for split in ("train", "val", "test"):
        X, y = z[f"X_{split}"], z[f"y_{split}"]
        if as_tensors:
            X = to_nchw(X)
            if normalize:
                mean = torch.tensor(SHAPE_MEAN).view(1, 3, 1, 1)
                std = torch.tensor(SHAPE_STD).view(1, 3, 1, 1)
                X = (X - mean) / std
            y = torch.from_numpy(y).long()
        out[split] = (X, y)
    return out


def load_digits_npz(as_tensors: bool = True):
    """scikit-learn's 8x8 handwritten digits, pre-split."""
    z = np.load(_require("digits_8x8.npz"))
    Xtr, ytr, Xte, yte = z["X_train"], z["y_train"], z["X_test"], z["y_test"]
    if as_tensors:
        Xtr = torch.from_numpy(Xtr).float().unsqueeze(1)
        Xte = torch.from_numpy(Xte).float().unsqueeze(1)
        ytr = torch.from_numpy(ytr).long()
        yte = torch.from_numpy(yte).long()
    return {"train": (Xtr, ytr), "test": (Xte, yte), "classes": [str(i) for i in range(10)]}


def load_detection():
    """96x96 multi-object scenes with bounding boxes."""
    z = np.load(_require("shapes_det.npz"), allow_pickle=True)
    ann = json.loads(_require("shapes_det_annotations.json").read_text(encoding="utf-8"))
    return {
        "train": (z["X_train"], ann["train"]),
        "val": (z["X_val"], ann["val"]),
        "classes": [str(c) for c in z["classes"]],
    }


def load_segmentation(as_tensors: bool = True):
    """Same scenes, with a per-pixel class mask (0 = background)."""
    z = np.load(_require("shapes_seg.npz"), allow_pickle=True)
    out = {"classes": ["background"] + [str(c) for c in z["classes"]]}
    for split in ("train", "val"):
        X, M = z[f"X_{split}"], z[f"M_{split}"]
        if as_tensors:
            X = to_nchw(X)
            M = torch.from_numpy(M).long()
        out[split] = (X, M)
    return out


def load_ssl_pool(as_tensors: bool = True):
    """Unlabelled pool for self-supervised pre-training.

    ``y_hidden`` exists only so you can measure linear-probe accuracy at the
    end -- it must not be used during pre-training.
    """
    z = np.load(_require("shapes_pairs.npz"), allow_pickle=True)
    X, y = z["X"], z["y_hidden"]
    if as_tensors:
        X = to_nchw(X)
        y = torch.from_numpy(y).long()
    return {"X": X, "y_hidden": y, "classes": [str(c) for c in z["classes"]]}


def load_sentiment():
    """Binary sentiment corpus as (texts, labels)."""
    texts, labels = [], []
    with open(_require("sentiment.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            texts.append(row["text"])
            labels.append(int(row["label"]))
    return texts, np.array(labels, dtype=np.int64)


def load_corpus() -> str:
    """Raw text for character-level language modelling."""
    return _require("corpus.txt").read_text(encoding="utf-8")


def load_captions():
    """(image_index, caption) pairs aligned with shapes_32 train split."""
    rows = []
    with open(_require("captions.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append((int(row["image_index"]), row["caption"]))
    return rows


def make_loaders(train_ds, val_ds, batch_size=128, num_workers=0):
    """DataLoaders with Windows-safe defaults (num_workers=0 avoids spawn issues)."""
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )
