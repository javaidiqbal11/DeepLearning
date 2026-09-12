"""Generate every dataset the course needs, fully offline.

No internet, no torchvision downloads. Students clone the repo, run this once
(or use the pre-built files already committed) and every notebook works.

Datasets produced
-----------------
shapes_32.npz          32x32 RGB, 4 classes (circle/square/triangle/star)
shapes_imagefolder/    the same data as real PNG files in class folders
digits_8x8.npz         scikit-learn handwritten digits (real data, ships with sklearn)
shapes_det.npz         96x96 scenes + bounding boxes for detection
shapes_seg.npz         96x96 scenes + per-pixel masks for segmentation
shapes_pairs.npz       unlabelled images for self-supervised pre-training
sentiment.csv          small labelled text corpus
corpus.txt             character-level text corpus
captions.csv           image-caption pairs over the shapes images
"""
from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

CLASSES = ["circle", "square", "triangle", "star"]
RNG = np.random.default_rng(0)


# --------------------------------------------------------------------------
# primitive drawing
# --------------------------------------------------------------------------
def _poly_star(cx, cy, r, rot=0.0, points=5):
    pts = []
    for i in range(points * 2):
        rad = r if i % 2 == 0 else r * 0.42
        ang = rot + i * math.pi / points - math.pi / 2
        pts.append((cx + rad * math.cos(ang), cy + rad * math.sin(ang)))
    return pts


def _poly_triangle(cx, cy, r, rot=0.0):
    return [
        (cx + r * math.cos(rot + i * 2 * math.pi / 3 - math.pi / 2),
         cy + r * math.sin(rot + i * 2 * math.pi / 3 - math.pi / 2))
        for i in range(3)
    ]


def _poly_square(cx, cy, r, rot=0.0):
    return [
        (cx + r * math.cos(rot + i * math.pi / 2 + math.pi / 4),
         cy + r * math.sin(rot + i * math.pi / 2 + math.pi / 4))
        for i in range(4)
    ]


def _draw_shape(draw, kind, cx, cy, r, rot, fill, outline=None):
    """Draw one shape and return its axis-aligned bounding box."""
    if kind == "circle":
        box = [cx - r, cy - r, cx + r, cy + r]
        draw.ellipse(box, fill=fill, outline=outline)
        return box
    poly = {"square": _poly_square, "triangle": _poly_triangle, "star": _poly_star}[kind](cx, cy, r, rot)
    draw.polygon(poly, fill=fill, outline=outline)
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return [min(xs), min(ys), max(xs), max(ys)]


def _bg_colour(rng):
    """Muted background so the shape is the signal, not the brightness."""
    base = rng.integers(200, 250)
    return tuple(int(np.clip(base + rng.integers(-12, 12), 0, 255)) for _ in range(3))


def _fg_colour(rng):
    """Saturated foreground, deliberately decorrelated from the class label."""
    h = rng.random()
    r, g, b = [int(255 * c) for c in _hsv_to_rgb(h, 0.55 + 0.35 * rng.random(), 0.55 + 0.3 * rng.random())]
    return (r, g, b)


def _hsv_to_rgb(h, s, v):
    i = int(h * 6.0)
    f = h * 6.0 - i
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    return [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i % 6]


# --------------------------------------------------------------------------
# single-object classification set
# --------------------------------------------------------------------------
def make_shapes(n, size=32, seed=0, noise=True):
    rng = np.random.default_rng(seed)
    X = np.zeros((n, size, size, 3), dtype=np.uint8)
    y = np.zeros(n, dtype=np.int64)
    for i in range(n):
        label = int(rng.integers(0, len(CLASSES)))
        img = Image.new("RGB", (size, size), _bg_colour(rng))
        d = ImageDraw.Draw(img)
        # Shapes fill most of the frame: small shapes at 32px are not separable
        # even for a CNN, which would make the lecture-5 result misleading.
        r = size * (0.30 + 0.14 * rng.random())
        cx = size / 2 + rng.normal(0, size * 0.05)
        cy = size / 2 + rng.normal(0, size * 0.05)
        cx = float(np.clip(cx, r, size - r))
        cy = float(np.clip(cy, r, size - r))
        _draw_shape(d, CLASSES[label], cx, cy, r, rng.random() * math.pi * 2, _fg_colour(rng))
        arr = np.asarray(img, dtype=np.int16)
        if noise:
            arr = arr + rng.normal(0, 6, arr.shape)
        X[i] = np.clip(arr, 0, 255).astype(np.uint8)
        y[i] = label
    return X, y


def build_classification():
    splits = {}
    for name, n, seed in [("train", 6000, 1), ("val", 1000, 2), ("test", 1000, 3)]:
        X, y = make_shapes(n, 32, seed)
        splits[f"X_{name}"], splits[f"y_{name}"] = X, y
    np.savez_compressed(DATA / "shapes_32.npz", classes=np.array(CLASSES), **splits)

    # A real on-disk image folder so students practise the file -> tensor path.
    root = DATA / "shapes_imagefolder"
    for split, n, seed in [("train", 800, 11), ("val", 200, 12)]:
        X, y = make_shapes(n, 64, seed)
        for ci, cname in enumerate(CLASSES):
            (root / split / cname).mkdir(parents=True, exist_ok=True)
        for i, (img, lab) in enumerate(zip(X, y)):
            Image.fromarray(img).save(root / split / CLASSES[lab] / f"{split}_{i:04d}.png")
    print(f"  shapes_32.npz  train={splits['X_train'].shape} val={splits['X_val'].shape}")
    print(f"  shapes_imagefolder/  1000 PNG files")


# --------------------------------------------------------------------------
# scikit-learn digits -> real handwritten data, offline
# --------------------------------------------------------------------------
def build_digits():
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    d = load_digits()
    X = (d.images / 16.0).astype(np.float32)   # 8x8, values 0..1
    y = d.target.astype(np.int64)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    np.savez_compressed(DATA / "digits_8x8.npz", X_train=Xtr, y_train=ytr, X_test=Xte, y_test=yte)
    print(f"  digits_8x8.npz  train={Xtr.shape} test={Xte.shape}")


# --------------------------------------------------------------------------
# detection + segmentation scenes
# --------------------------------------------------------------------------
def build_detection(n_train=1500, n_val=300, size=96):
    def scene(rng):
        img = Image.new("RGB", (size, size), _bg_colour(rng))
        d = ImageDraw.Draw(img)
        mask = Image.new("L", (size, size), 0)
        md = ImageDraw.Draw(mask)
        boxes, labels = [], []
        for _ in range(int(rng.integers(1, 4))):
            label = int(rng.integers(0, len(CLASSES)))
            r = size * (0.11 + 0.08 * rng.random())
            cx = float(rng.uniform(r + 1, size - r - 1))
            cy = float(rng.uniform(r + 1, size - r - 1))
            rot = rng.random() * math.pi * 2
            colour = _fg_colour(rng)
            box = _draw_shape(d, CLASSES[label], cx, cy, r, rot, colour)
            _draw_shape(md, CLASSES[label], cx, cy, r, rot, label + 1)   # class id in the mask
            boxes.append([max(0.0, box[0]), max(0.0, box[1]),
                          min(float(size), box[2]), min(float(size), box[3])])
            labels.append(label)
        return np.asarray(img, np.uint8), np.asarray(mask, np.uint8), boxes, labels

    out = {}
    ann = {"classes": CLASSES, "train": [], "val": []}
    for split, n, seed in [("train", n_train, 21), ("val", n_val, 22)]:
        rng = np.random.default_rng(seed)
        imgs, masks = [], []
        for i in range(n):
            im, mk, boxes, labels = scene(rng)
            imgs.append(im)
            masks.append(mk)
            ann[split].append({"id": i, "boxes": boxes, "labels": labels})
        out[f"X_{split}"] = np.stack(imgs)
        out[f"M_{split}"] = np.stack(masks)

    np.savez_compressed(DATA / "shapes_seg.npz", classes=np.array(CLASSES),
                        **{k: v for k, v in out.items()})
    np.savez_compressed(DATA / "shapes_det.npz", classes=np.array(CLASSES),
                        X_train=out["X_train"], X_val=out["X_val"])
    (DATA / "shapes_det_annotations.json").write_text(json.dumps(ann), encoding="utf-8")
    print(f"  shapes_det.npz / shapes_seg.npz  train={out['X_train'].shape}")


# --------------------------------------------------------------------------
# unlabelled pool for self-supervised learning
# --------------------------------------------------------------------------
def build_ssl():
    X, y = make_shapes(4000, 32, seed=31)
    np.savez_compressed(DATA / "shapes_pairs.npz", X=X, y_hidden=y, classes=np.array(CLASSES))
    print(f"  shapes_pairs.npz  X={X.shape} (labels withheld as y_hidden)")


# --------------------------------------------------------------------------
# text corpora
# --------------------------------------------------------------------------
POS_OPEN = ["absolutely", "really", "genuinely", "honestly", "truly", "pretty"]
POS_ADJ = ["brilliant", "excellent", "delightful", "solid", "impressive", "charming",
           "gripping", "moving", "clever", "beautiful"]
NEG_ADJ = ["dreadful", "boring", "predictable", "shallow", "clumsy", "tedious",
           "disappointing", "muddled", "lifeless", "forgettable"]
NOUNS = ["film", "story", "script", "cast", "soundtrack", "pacing", "ending",
         "cinematography", "dialogue", "performance"]
POS_TAIL = ["I would watch it again.", "Worth every minute.", "A real surprise.",
            "Highly recommended.", "It stayed with me for days."]
NEG_TAIL = ["I wanted the time back.", "Hard to sit through.", "Skip it.",
            "A waste of a good cast.", "It never recovers."]


def build_text():
    rnd = random.Random(7)
    rows = []
    for _ in range(3000):
        pos = rnd.random() < 0.5
        adj = rnd.choice(POS_ADJ if pos else NEG_ADJ)
        text = (f"The {rnd.choice(NOUNS)} was {rnd.choice(POS_OPEN)} {adj}. "
                f"The {rnd.choice(NOUNS)} felt {rnd.choice(POS_ADJ if pos else NEG_ADJ)}. "
                f"{rnd.choice(POS_TAIL if pos else NEG_TAIL)}")
        rows.append({"text": text, "label": int(pos)})
    rnd.shuffle(rows)
    with open(DATA / "sentiment.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["text", "label"])
        w.writeheader()
        w.writerows(rows)

    corpus = "\n".join(r["text"] for r in rows)
    (DATA / "corpus.txt").write_text(corpus, encoding="utf-8")

    # image-caption pairs that describe the shapes dataset
    z = np.load(DATA / "shapes_32.npz", allow_pickle=True)
    caps = []
    templates = ["a {c} in the middle of the frame", "a small {c} on a pale background",
                 "one {c} shape", "picture of a {c}", "a single {c}, centred"]
    for i, lab in enumerate(z["y_train"][:2000]):
        caps.append({"image_index": int(i), "caption": rnd.choice(templates).format(c=CLASSES[lab])})
    with open(DATA / "captions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_index", "caption"])
        w.writeheader()
        w.writerows(caps)
    print(f"  sentiment.csv ({len(rows)} rows)  corpus.txt  captions.csv")


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    print("Building course datasets ->", DATA)
    build_classification()
    build_digits()
    build_detection()
    build_ssl()
    build_text()
    print("Done.")


if __name__ == "__main__":
    main()
