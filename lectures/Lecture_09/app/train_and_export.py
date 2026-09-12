"""Train the shape classifier and export a complete deployable artefact.

    python train_and_export.py

This is the "offline" half of the system. It deliberately writes the
preprocessing configuration into the same file as the weights — that pairing is
what makes the artefact deployable rather than merely saved.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(APP_DIR))

from dlcourse import load_shapes, train, evaluate, set_seed, count_parameters  # noqa: E402
from dlcourse.data import SHAPE_MEAN, SHAPE_STD                                # noqa: E402
from model import ModelBundle, PreprocessConfig, ShapeClassifier               # noqa: E402

VERSION = "1.0.0"
EPOCHS = 6


def main():
    set_seed(0)
    data = load_shapes(normalize=True)
    X_train, y_train = data["train"]
    X_val, y_val = data["val"]
    X_test, y_test = data["test"]
    classes = data["classes"]

    model = ShapeClassifier(n_classes=len(classes))
    print(f"training {count_parameters(model):,} parameters for {EPOCHS} epochs")

    t0 = time.time()
    train(model, (X_train, y_train), (X_val, y_val),
          epochs=EPOCHS, lr=2e-3, batch_size=128)
    train_seconds = time.time() - t0

    test_loss, test_acc = evaluate(model, (X_test, y_test))
    print(f"\ntest accuracy: {test_acc:.4f}  (loss {test_loss:.4f})")

    preprocess = PreprocessConfig(input_size=32, mean=SHAPE_MEAN, std=SHAPE_STD)
    metrics = {
        "test_accuracy": round(test_acc, 4),
        "test_loss": round(test_loss, 4),
        "train_seconds": round(train_seconds, 1),
        "epochs": EPOCHS,
        "n_train": len(X_train),
    }
    path = ModelBundle.save(model, classes, preprocess, VERSION, metrics)
    print(f"\nartefact written to {path}")
    print(f"  size: {path.stat().st_size / 1024:.0f} KB")
    print(f"  contents: state_dict, classes, preprocess config, version, metrics")

    # Prove the artefact round-trips before declaring success.
    bundle = ModelBundle.load(path)
    probe = torch.zeros(1, 3, 32, 32)
    probs, ms = bundle.predict(probe)
    print(f"\nreload check: version {bundle.version}, "
          f"output shape {probs.shape}, {ms:.2f} ms")

    # Write a few sample images the API tests and the notebook can post.
    sample_dir = APP_DIR / "samples"
    sample_dir.mkdir(exist_ok=True)
    from PIL import Image
    raw = load_shapes()["test"][0]
    for c in range(len(classes)):
        idx = int((y_test == c).nonzero()[0])
        arr = (raw[idx].permute(1, 2, 0).numpy() * 255).round().astype("uint8")
        Image.fromarray(arr).save(sample_dir / f"{classes[c]}.png")
    print(f"sample images written to {sample_dir}")


if __name__ == "__main__":
    main()
