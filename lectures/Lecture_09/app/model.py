"""Model definition, artefact loading and inference.

The point of this module: a deployable artefact is the weights **plus** the
preprocessing that produced them. Shipping a `state_dict` alone is how training
and serving quietly drift apart.
"""
from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ARTEFACT_DIR = Path(__file__).resolve().parent / "artefacts"
CHECKPOINT = ARTEFACT_DIR / "shape_classifier.pt"


# --------------------------------------------------------------------------
# architecture
# --------------------------------------------------------------------------
class BasicBlock(nn.Module):
    """ResNet basic block — must match the definition used at training time."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ShapeClassifier(nn.Module):
    def __init__(self, n_classes: int = 4, widths=(16, 32, 64), blocks_per_stage: int = 2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, widths[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]), nn.ReLU(),
        )
        stages, in_ch = [], widths[0]
        for i, w in enumerate(widths):
            for b in range(blocks_per_stage):
                stages.append(BasicBlock(in_ch, w, stride=2 if (b == 0 and i > 0) else 1))
                in_ch = w
        self.stages = nn.Sequential(*stages)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Linear(in_ch, n_classes))

    def forward(self, x):
        return self.head(self.stages(self.stem(x)))


# --------------------------------------------------------------------------
# the artefact
# --------------------------------------------------------------------------
@dataclass
class PreprocessConfig:
    """Everything needed to turn an uploaded file into a model input.

    Stored with the weights, because a mismatch here is silent: the service
    returns confident, wrong answers and no error is ever raised.
    """
    input_size: int = 32
    mean: tuple[float, float, float] = (0.878, 0.874, 0.876)
    std: tuple[float, float, float] = (0.168, 0.176, 0.180)
    resample: str = "bilinear"


class ModelBundle:
    """A loaded model together with its preprocessing and metadata."""

    def __init__(self, model: nn.Module, classes: list[str],
                 preprocess: PreprocessConfig, version: str, metrics: dict):
        self.model = model.eval()
        self.classes = classes
        self.preprocess = preprocess
        self.version = version
        self.metrics = metrics
        self._mean = torch.tensor(preprocess.mean).view(1, 3, 1, 1)
        self._std = torch.tensor(preprocess.std).view(1, 3, 1, 1)

    # -- loading -----------------------------------------------------------
    @classmethod
    def load(cls, path: Path = CHECKPOINT) -> "ModelBundle":
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run `python app/train_and_export.py` first."
            )
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = ShapeClassifier(n_classes=len(ckpt["classes"]))
        model.load_state_dict(ckpt["state_dict"])
        return cls(
            model=model,
            classes=ckpt["classes"],
            preprocess=PreprocessConfig(**ckpt["preprocess"]),
            version=ckpt["version"],
            metrics=ckpt.get("metrics", {}),
        )

    @staticmethod
    def save(model: nn.Module, classes: list[str], preprocess: PreprocessConfig,
             version: str, metrics: dict, path: Path = CHECKPOINT) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": model.state_dict(),
            "classes": classes,
            "preprocess": asdict(preprocess),
            "version": version,
            "metrics": metrics,
        }, path)
        return path

    # -- preprocessing -----------------------------------------------------
    def bytes_to_tensor(self, raw: bytes) -> torch.Tensor:
        """Decode an uploaded file into a normalised 1x3xSxS tensor.

        Raises ValueError on anything that is not a decodable image — the API
        layer turns that into a 422.
        """
        try:
            img = Image.open(io.BytesIO(raw))
            img.load()                       # force decode now, not lazily
        except Exception as e:
            raise ValueError(f"could not decode image: {e}") from e

        img = img.convert("RGB")
        size = self.preprocess.input_size
        if img.size != (size, size):
            img = img.resize((size, size), Image.BILINEAR)

        arr = np.asarray(img, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return (t - self._mean) / self._std

    # -- inference ---------------------------------------------------------
    @torch.inference_mode()
    def predict(self, batch: torch.Tensor) -> tuple[np.ndarray, float]:
        """Returns (probabilities [N, C], inference milliseconds)."""
        t0 = time.perf_counter()
        logits = self.model(batch)
        probs = torch.softmax(logits, dim=1).numpy()
        return probs, (time.perf_counter() - t0) * 1000.0

    def top_prediction(self, probs_row: np.ndarray) -> tuple[str, float]:
        idx = int(np.argmax(probs_row))
        return self.classes[idx], float(probs_row[idx])

    def as_dict(self, probs_row: np.ndarray) -> dict[str, float]:
        return {c: round(float(p), 6) for c, p in zip(self.classes, probs_row)}

    def warmup(self, n: int = 3) -> None:
        """The first inference is always slower. Pay that cost at startup."""
        dummy = torch.zeros(1, 3, self.preprocess.input_size, self.preprocess.input_size)
        for _ in range(n):
            self.predict(dummy)
