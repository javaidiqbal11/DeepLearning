"""A train/test split with a deliberate flaw. Find it. (Lecture 1, stretch task.)

Run me:
    python lectures/Lecture_01/leaky_split.py

The reported accuracy looks excellent. It is not real. Your job is to explain
why, measure how much of that number is fake, and write an honest version.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sklearn.neighbors import KNeighborsClassifier   # noqa: E402

N_ORIGINALS = 2000
COPIES = 3


def augment(X: np.ndarray, y: np.ndarray, copies: int = COPIES):
    """Enlarge the dataset with small random shifts of each image."""
    rng = np.random.default_rng(0)
    out_X, out_y = [X], [y]
    for _ in range(copies):
        shifted = np.empty_like(X)
        for i in range(len(X)):
            dx, dy = rng.integers(-2, 3, size=2)
            shifted[i] = np.roll(X[i], (int(dx), int(dy)), axis=(0, 1))
        out_X.append(shifted)
        out_y.append(y)
    return np.concatenate(out_X), np.concatenate(out_y)


def build_split():
    z = np.load(ROOT / "data" / "shapes_32.npz", allow_pickle=True)
    X, y = z["X_train"][:N_ORIGINALS], z["y_train"][:N_ORIGINALS]

    # Step 1: grow the dataset by augmenting.
    X_big, y_big = augment(X, y)

    # Step 2: shuffle everything together and split 80/20.
    rng = np.random.default_rng(1)
    idx = rng.permutation(len(X_big))
    cut = int(0.8 * len(idx))
    tr, te = idx[:cut], idx[cut:]

    flat = X_big.reshape(len(X_big), -1).astype(np.float32) / 255.0
    return flat[tr], y_big[tr], flat[te], y_big[te]


def main():
    Xtr, ytr, Xte, yte = build_split()
    print(f"train {Xtr.shape}   test {Xte.shape}")

    model = KNeighborsClassifier(n_neighbors=1).fit(Xtr, ytr)
    acc = model.score(Xte, yte)

    print(f"\nreported test accuracy: {acc:.3f}")
    print("\nThe same 1-NN classifier trained on 2000 clean images and tested on the")
    print("genuine held-out test split reaches about 0.86. This split reports 0.96.")
    print("Roughly ten points of that number are fake. The answer is in build_split().")
    print("\nYour tasks:")
    print("  1. Name the flaw in one sentence.")
    print("  2. Write build_split_honest() that fixes it.")
    print("  3. Report both accuracies and the size of the inflation.")


if __name__ == "__main__":
    main()
