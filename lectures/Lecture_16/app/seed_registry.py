"""Train two model versions and register them, so the service has something to serve.

    python seed_registry.py

Deliberately registers a weak v0.9.0 and a better v1.0.0, so students can
exercise promotion and rollback against a real quality difference.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(ROOT / "lectures" / "Lecture_09" / "app"))

from dlcourse import load_shapes, train, evaluate, set_seed             # noqa: E402
from dlcourse.data import SHAPE_MEAN, SHAPE_STD                         # noqa: E402
from model import PreprocessConfig, ShapeClassifier                     # noqa: E402
from registry import ModelRegistry                                      # noqa: E402


@torch.no_grad()
def prediction_distribution(model, X, n_classes):
    model.eval()
    preds = torch.cat([model(X[i:i + 256]).argmax(1) for i in range(0, len(X), 256)])
    return (torch.bincount(preds, minlength=n_classes).float() / len(preds)).tolist()


def build(n_train: int, epochs: int, seed: int = 0):
    data = load_shapes(normalize=True)
    X_train, y_train = data["train"]
    X_val, y_val = data["val"]
    X_test, y_test = data["test"]

    set_seed(seed)
    model = ShapeClassifier(n_classes=len(data["classes"]))
    train(model, (X_train[:n_train], y_train[:n_train]), (X_val, y_val),
          epochs=epochs, lr=2e-3, batch_size=128, verbose=False)
    loss, acc = evaluate(model, (X_test, y_test))
    baseline = prediction_distribution(model, X_test, len(data["classes"]))
    return model, data["classes"], acc, loss, baseline


def main():
    registry = ModelRegistry()
    preprocess = PreprocessConfig(input_size=32, mean=SHAPE_MEAN, std=SHAPE_STD)

    specs = [
        ("0.9.0", 400, 3, "first attempt: small training subset"),
        ("1.0.0", 6000, 5, "full training set"),
    ]

    for version, n_train, epochs, notes in specs:
        if registry.get(version):
            print(f"{version} already registered, skipping")
            continue
        print(f"training {version} ({n_train} images, {epochs} epochs)...")
        model, classes, acc, loss, baseline = build(n_train, epochs)
        registry.register(
            payload={"state_dict": model.state_dict(), "classes": classes,
                     "preprocess": preprocess.__dict__},
            version=version,
            metrics={"test_accuracy": round(acc, 4), "test_loss": round(loss, 4)},
            config={"n_train": n_train, "epochs": epochs, "lr": 2e-3},
            notes=notes,
            baseline_distribution=[round(b, 4) for b in baseline],
        )
        print(f"  registered {version}: test accuracy {acc:.4f}")

    registry.promote("1.0.0")
    print("\npromoted 1.0.0 to production")
    print(f"\n{'version':<10}{'stage':<12}{'test acc':>10}   notes")
    print("-" * 58)
    for rec in registry.list_models():
        print(f"{rec['version']:<10}{rec['stage']:<12}"
              f"{rec['metrics']['test_accuracy']:>10.4f}   {rec['notes']}")

    print("\nNow run:  uvicorn main:app --reload")


if __name__ == "__main__":
    main()
