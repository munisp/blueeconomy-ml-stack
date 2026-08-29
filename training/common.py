"""Shared training utilities: seeds, metrics, tracking, early stopping."""

from __future__ import annotations

import contextlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np

try:  # MLflow is the primary tracker; degrade gracefully to local JSON logs.
    import mlflow
    _HAS_MLFLOW = True
except ImportError:  # pragma: no cover - environment-dependent
    mlflow = None
    _HAS_MLFLOW = False

RESULTS_DIR = Path(os.environ.get("BEML_RESULTS_DIR", "results"))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def get_device(name: str = "cpu") -> "object":
    import torch
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "cuda":
        print("[warn] --device cuda requested but CUDA unavailable; using CPU")
    return torch.device("cpu")


class EarlyStopping:
    def __init__(self, patience: int = 8, min_delta: float = 1e-4):
        self.patience, self.min_delta = patience, min_delta
        self.best, self.wait, self.stop = -np.inf, 0, False

    def step(self, metric: float) -> bool:
        if metric > self.best + self.min_delta:
            self.best, self.wait = metric, 0
        else:
            self.wait += 1
            self.stop = self.wait >= self.patience
        return self.stop


class RunTracker:
    """MLflow run when available; otherwise append a JSON-lines run log.

    The interface is deliberately tiny (params/metrics/artifact) so both
    backends stay honest: every metric reported in MODEL_CARDS.md is written
    either to MLflow or to results/runs.jsonl by the same code path.
    """

    def __init__(self, experiment: str, run_name: str):
        self.experiment, self.run_name = experiment, run_name
        self._record = {"experiment": experiment, "run_name": run_name,
                        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "params": {}, "metrics": {}, "backend": "jsonl"}
        self._ctx = contextlib.nullcontext()
        if _HAS_MLFLOW:
            mlflow.set_experiment(experiment)
            self._ctx = mlflow.start_run(run_name=run_name)
            self._record["backend"] = "mlflow"

    def __enter__(self) -> "RunTracker":
        self._ctx.__enter__()
        return self

    def __exit__(self, *exc) -> None:
        self._ctx.__exit__(*exc)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_DIR / "runs.jsonl", "a") as f:
            f.write(json.dumps(self._record) + "\n")

    def log_params(self, params: dict) -> None:
        self._record["params"].update({k: str(v) for k, v in params.items()})
        if _HAS_MLFLOW:
            mlflow.log_params({k: str(v) for k, v in params.items()})

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        self._record["metrics"].update({k: float(v) for k, v in metrics.items()})
        if _HAS_MLFLOW:
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)

    def log_artifact(self, path: str | Path) -> None:
        if _HAS_MLFLOW:
            mlflow.log_artifact(str(path))


def classification_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    from sklearn.metrics import (average_precision_score, precision_recall_curve,
                                 roc_auc_score)
    out = {
        "auroc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
    }
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    # recall at precision >= 0.90 (operating doctrine: high-precision alerts)
    mask = prec[:-1] >= 0.90
    out["recall_at_precision_0.90"] = float(rec[:-1][mask].max()) if mask.any() else 0.0
    # best F1 threshold
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-9, None)
    i = int(np.argmax(f1))
    out["best_f1"] = float(f1[i])
    out["best_f1_threshold"] = float(thr[i])
    return out
