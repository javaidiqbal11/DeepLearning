"""One training loop, reused by every lecture.

Notebooks that are *about* the training loop (lectures 3 and 4) write their own
from scratch; the rest import this so the new idea stays in focus.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int = 0) -> None:
    """Make a run reproducible. Call this first in every notebook."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@dataclass
class History:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    epoch_time: list[float] = field(default_factory=list)

    def best(self) -> tuple[int, float]:
        """(epoch index, best validation accuracy)."""
        if not self.val_acc:
            return -1, float("nan")
        i = int(np.argmax(self.val_acc))
        return i, self.val_acc[i]


def _as_loader(data, batch_size, shuffle):
    if isinstance(data, DataLoader):
        return data
    if isinstance(data, (tuple, list)) and len(data) == 2:
        return DataLoader(TensorDataset(*data), batch_size=batch_size, shuffle=shuffle)
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def evaluate(model, data, loss_fn=None, batch_size=256, dev=None):
    """Return (mean loss, accuracy). Works for classification heads."""
    dev = dev or device()
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    model.eval().to(dev)
    loader = _as_loader(data, batch_size, shuffle=False)
    total_loss, correct, n = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(dev), yb.to(dev)
        out = model(xb)
        total_loss += loss_fn(out, yb).item() * len(xb)
        correct += (out.argmax(1) == yb).sum().item()
        n += len(xb)
    return total_loss / max(n, 1), correct / max(n, 1)


def train(model, train_data, val_data=None, epochs=10, lr=1e-3, batch_size=128,
          loss_fn=None, optimizer=None, scheduler=None, dev=None, verbose=True,
          grad_clip=None):
    """Standard supervised loop. Returns a History.

    ``train_data`` / ``val_data`` may be a DataLoader, a Dataset, or an
    ``(X, y)`` tensor pair.
    """
    dev = dev or device()
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=lr)
    model.to(dev)

    train_loader = _as_loader(train_data, batch_size, shuffle=True)
    hist = History()

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        running, correct, n = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            optimizer.zero_grad(set_to_none=True)
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            running += loss.item() * len(xb)
            correct += (out.argmax(1) == yb).sum().item()
            n += len(xb)

        if scheduler is not None:
            scheduler.step()

        hist.train_loss.append(running / n)
        hist.train_acc.append(correct / n)
        hist.epoch_time.append(time.time() - t0)

        msg = f"epoch {epoch + 1:3d}/{epochs}  train_loss {hist.train_loss[-1]:.4f}  train_acc {hist.train_acc[-1]:.3f}"
        if val_data is not None:
            vl, va = evaluate(model, val_data, loss_fn, dev=dev)
            hist.val_loss.append(vl)
            hist.val_acc.append(va)
            msg += f"  val_loss {vl:.4f}  val_acc {va:.3f}"
        msg += f"  ({hist.epoch_time[-1]:.1f}s)"
        if verbose:
            print(msg)

    return hist
