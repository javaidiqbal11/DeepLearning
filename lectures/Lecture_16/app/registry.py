"""Model registry: versioned artefacts with an explicit promotion step.

The rule this enforces: exactly one version is in production at a time, every
version keeps its metrics and config, and nothing is ever deleted on promotion —
so rollback is always one call away.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import torch

REGISTRY_DIR = Path(__file__).resolve().parent / "registry_store"


@dataclass
class ModelRecord:
    version: str
    created_at: str
    metrics: dict
    config: dict
    stage: str = "staging"          # staging -> production -> archived
    notes: str = ""
    baseline_distribution: list[float] = field(default_factory=list)


class ModelRegistry:
    def __init__(self, root: Path = REGISTRY_DIR):
        self.root = Path(root)
        self.artefacts = self.root / "artefacts"
        self.artefacts.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"
        if not self.index_path.exists():
            self.index_path.write_text("{}", encoding="utf-8")

    # -- storage -----------------------------------------------------------
    def _read(self) -> dict:
        return json.loads(self.index_path.read_text(encoding="utf-8"))

    def _write(self, index: dict) -> None:
        self.index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")

    def artefact_path(self, version: str) -> Path:
        return self.artefacts / f"{version}.pt"

    # -- operations --------------------------------------------------------
    def register(self, payload: dict, version: str, metrics: dict, config: dict,
                 notes: str = "", baseline_distribution: list[float] | None = None) -> str:
        """``payload`` is whatever ``torch.save`` should write — normally a dict
        containing the state_dict, class names and preprocessing config."""
        index = self._read()
        if version in index:
            raise ValueError(f"version {version} already registered; versions are immutable")

        torch.save(payload, self.artefact_path(version))
        index[version] = asdict(ModelRecord(
            version=version,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            metrics=metrics,
            config=config,
            notes=notes,
            baseline_distribution=baseline_distribution or [],
        ))
        self._write(index)
        return version

    def promote(self, version: str) -> dict:
        index = self._read()
        if version not in index:
            raise KeyError(version)
        for rec in index.values():
            if rec["stage"] == "production":
                rec["stage"] = "archived"
        index[version]["stage"] = "production"
        self._write(index)
        return index[version]

    def production(self) -> dict | None:
        for rec in self._read().values():
            if rec["stage"] == "production":
                return rec
        return None

    def get(self, version: str) -> dict | None:
        return self._read().get(version)

    def list_models(self) -> list[dict]:
        return sorted(self._read().values(), key=lambda r: r["created_at"], reverse=True)

    def load_payload(self, version: str) -> dict:
        path = self.artefact_path(version)
        if not path.exists():
            raise FileNotFoundError(path)
        return torch.load(path, map_location="cpu", weights_only=False)
