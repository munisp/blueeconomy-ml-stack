"""Fail-closed ONNX scoring library (CPU, onnxruntime).

A Scorer wraps one versioned ONNX model directory:

    models/<model-name>/<semver>/model.onnx + metrics.json

Fail-closed contract:
- model file missing, unreadable, or failing ONNX validation  -> UNAVAILABLE
- input feature count mismatch                                -> UNAVAILABLE
- never, under any circumstance, return a fabricated score

Model-version routing for A/B is a deterministic hash split on the entity ID
(same entity always lands on the same version) — see monitoring/ab.py for the
shared implementation consumed by both this service and offline analysis.

CPU latency budget (measured on the tiny committed models, see README):
p50 single-score latency is expected in the low single-digit milliseconds;
the service enforces a configurable per-request budget and reports
SCORING_UNAVAILABLE on budget breach rather than silently returning late.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

STATUS_OK = "OK"
STATUS_UNAVAILABLE = "SCORING_UNAVAILABLE"


class ModelUnavailableError(RuntimeError):
    """Raised when a model cannot be loaded or executed. Fail-closed signal."""


class NonFiniteScoreError(ValueError):
    """Raised when input features or a model output are NaN/Inf. A non-finite
    value must never be served as status=OK (it would serialize as invalid
    strict JSON into both the API response and the signed inference event)."""


@dataclass
class ScoreResult:
    status: str                       # STATUS_OK | STATUS_UNAVAILABLE
    score: float | None               # None when unavailable — never fabricated
    model_name: str
    model_version: str | None
    mode: str                         # "ml" | "rules_only"
    latency_ms: float
    detail: str = ""


@dataclass
class _LoadedModel:
    session: "object"                 # onnxruntime.InferenceSession
    n_features: int
    version: str
    kind: str                         # "classifier" | "autoencoder" | "gnn"


class Scorer:
    """Thread-safe scorer over a versioned model root with A/B routing."""

    def __init__(self, models_root: str | Path, model_name: str,
                 versions: list[str], split: list[float] | None = None,
                 latency_budget_ms: float = 50.0,
                 salt: str = "blueeconomy-ab-v1"):
        from monitoring.ab import HashSplitter
        self.models_root = Path(models_root)
        self.model_name = model_name
        self.versions = versions
        self.splitter = HashSplitter(
            versions, split or [1.0 / len(versions)] * len(versions),
            salt=salt)
        self.latency_budget_ms = latency_budget_ms
        self._cache: dict[str, _LoadedModel | None] = {}
        self._lock = threading.Lock()

    # ---- loading (fail-closed) ----
    def _load(self, version: str) -> _LoadedModel | None:
        if version in self._cache:
            return self._cache[version]
        with self._lock:
            if version in self._cache:
                return self._cache[version]
            loaded = None
            try:
                import onnxruntime as ort
                path = self.models_root / self.model_name / version / "model.onnx"
                if not path.is_file():
                    raise ModelUnavailableError(f"missing model file: {path}")
                so = ort.SessionOptions()
                so.inter_op_num_threads = 1
                so.intra_op_num_threads = max(1, min(4, __import__("os").cpu_count() or 1))
                session = ort.InferenceSession(str(path), sess_options=so,
                                               providers=["CPUExecutionProvider"])
                meta_path = path.parent / "metrics.json"
                kind = "classifier"
                n_features = session.get_inputs()[0].shape[-1]
                if meta_path.is_file():
                    meta = json.loads(meta_path.read_text())
                    kind = meta.get("kind", kind)
                if not isinstance(n_features, int) or n_features <= 0:
                    raise ModelUnavailableError(f"invalid input shape for {path}")
                loaded = _LoadedModel(session=session, n_features=int(n_features),
                                      version=version, kind=kind)
            except Exception as exc:  # fail closed: cache the failure
                loaded = None
                self._last_error = str(exc)
            self._cache[version] = loaded
            return loaded

    # ---- warmup ----
    def warmup(self) -> dict[str, bool]:
        """Eagerly load every registered version (same fail-closed path as
        the first score call). Returns {version: loaded?}. A version that
        fails to load stays cached as a failure, so warmup errors are
        honestly surfaced by later score() calls as SCORING_UNAVAILABLE —
        warmup never fabricates availability."""
        import logging
        result = {}
        for version in self.versions:
            loaded = self._load(version)
            result[version] = loaded is not None
            if loaded is None:
                logging.getLogger(__name__).warning(
                    "warmup failed for %s@%s: %s", self.model_name, version,
                    getattr(self, "_last_error", "unknown"))
        return result

    # ---- scoring ----
    def score(self, features: list[float], entity_id: str = "") -> ScoreResult:
        version = self.splitter.route(entity_id) if entity_id else self.versions[0]
        # Model load is outside the per-request latency budget (budget covers
        # feature validation + ONNX execution only; cold loads are a deploy
        # concern handled by pre-warming).
        model = self._load(version)
        t0 = time.perf_counter()
        if model is None:
            return ScoreResult(status=STATUS_UNAVAILABLE, score=None,
                               model_name=self.model_name, model_version=version,
                               mode="rules_only",
                               latency_ms=self._elapsed(t0),
                               detail=getattr(self, "_last_error", "model unavailable"))
        x = np.asarray(features, dtype=np.float32).reshape(1, -1)
        if not np.isfinite(x).all():
            # Refuse NaN/Inf inputs: the model would propagate them into a
            # non-finite score, which must never be served as status=OK.
            return ScoreResult(status=STATUS_UNAVAILABLE, score=None,
                               model_name=self.model_name, model_version=version,
                               mode="rules_only", latency_ms=self._elapsed(t0),
                               detail="NON_FINITE_INPUT: features contain NaN/Inf")
        if x.shape[1] != model.n_features:
            return ScoreResult(status=STATUS_UNAVAILABLE, score=None,
                               model_name=self.model_name, model_version=version,
                               mode="rules_only", latency_ms=self._elapsed(t0),
                               detail=f"feature count {x.shape[1]} != model expects "
                                      f"{model.n_features}")
        try:
            raw = float(model.session.run(None, {"features": x})[0].reshape(-1)[0])
        except Exception as exc:
            return ScoreResult(status=STATUS_UNAVAILABLE, score=None,
                               model_name=self.model_name, model_version=version,
                               mode="rules_only", latency_ms=self._elapsed(t0),
                               detail=f"onnx runtime failure: {exc}")
        if not np.isfinite(raw):
            # Fail closed on a NaN/Inf model output: serving it as OK would
            # emit invalid strict JSON and poison the signed inference event.
            return ScoreResult(status=STATUS_UNAVAILABLE, score=None,
                               model_name=self.model_name, model_version=version,
                               mode="rules_only", latency_ms=self._elapsed(t0),
                               detail="NON_FINITE_OUTPUT: model produced NaN/Inf")
        score = self._postprocess(raw, model.kind)
        latency = self._elapsed(t0)
        if latency > self.latency_budget_ms:
            return ScoreResult(status=STATUS_UNAVAILABLE, score=None,
                               model_name=self.model_name, model_version=version,
                               mode="rules_only", latency_ms=latency,
                               detail=f"latency {latency:.1f}ms exceeded budget "
                                      f"{self.latency_budget_ms:.1f}ms")
        return ScoreResult(status=STATUS_OK, score=score, model_name=self.model_name,
                           model_version=version, mode="ml", latency_ms=latency)

    @staticmethod
    def _postprocess(raw: float, kind: str) -> float:
        if kind == "classifier":
            return float(1.0 / (1.0 + np.exp(-raw)))  # logit -> probability
        # autoencoder: reconstruction error; regressor/policy: raw value
        # (for "policy" the raw value IS the recommended action index;
        # shadow-mode wrapping lives in inference/service.py)
        return float(raw)

    @staticmethod
    def _elapsed(t0: float) -> float:
        return (time.perf_counter() - t0) * 1000.0
