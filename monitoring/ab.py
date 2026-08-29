"""Deterministic hash-based A/B traffic splitting.

The same entity ID must always route to the same model version (stable
bucketing), on every service replica and in offline analysis, without any
shared state. Split is over SHA-256 of (salt, entity_id) — used purely for
deterministic bucketing, never as a signal.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class HashSplitter:
    versions: list[str]
    weights: list[float]
    salt: str = "blueeconomy-ab-v1"

    def __post_init__(self) -> None:
        if len(self.versions) != len(self.weights):
            raise ValueError("versions and weights must align")
        total = sum(self.weights)
        if total <= 0:
            raise ValueError("split weights must sum to a positive value")
        self._cum = []
        acc = 0.0
        for w in self.weights:
            acc += w / total
            self._cum.append(acc)

    def route(self, entity_id: str) -> str:
        digest = hashlib.sha256(f"{self.salt}|{entity_id}".encode()).hexdigest()
        u = int(digest[:12], 16) / float(0xFFFFFFFFFFFF)
        for version, edge in zip(self.versions, self._cum):
            if u <= edge:
                return version
        return self.versions[-1]

    @classmethod
    def from_config(cls, path: str | Path, model_name: str) -> "HashSplitter":
        """Load split config (JSON or simple YAML) consumed by the service."""
        text = Path(path).read_text()
        try:
            cfg = json.loads(text)
        except json.JSONDecodeError:
            import yaml  # pyyaml, permissive
            cfg = yaml.safe_load(text)
        entry = cfg["models"][model_name]
        return cls(versions=entry["versions"], weights=entry["weights"],
                   salt=cfg.get("salt", "blueeconomy-ab-v1"))
