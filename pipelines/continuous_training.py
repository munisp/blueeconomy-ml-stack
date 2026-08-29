"""Continuous training orchestrator.

Loop: extract snapshot -> retrain candidate -> evaluation gate (candidate must
beat the deployed model on held-out metrics) -> registry stage transition
(Staging -> Production) -> ONNX export -> notify.

The evaluation gate is the safety property: a model that does not improve on
held-out AUROC by at least --min-delta is NEVER promoted. Notifications are
written to a JSONL outbox (webhook wiring is a deploy concern).

Usage:
    python -m pipelines.continuous_training --model declaration-fraud \
        --data-dir data/synthetic --candidate-version 0.2.0
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

MODEL_SPECS = {
    "declaration-fraud": {
        "train_module": "training.tabular",
        "data_arg": "--data", "data_file": "declarations.parquet",
        "gate_metric": "test.auroc",
    },
    "vessel-anomaly": {
        "train_module": "training.anomaly",
        "data_arg": "--data", "data_file": "ais.parquet",
        "gate_metric": "test.auroc",
    },
    "graph-mule-gnn": {
        "train_module": "training.gnn",
        "data_arg": "--data-dir", "data_file": None,
        "gate_metric": "test.auroc",
    },
}


def _get_metric(metrics: dict, dotted: str) -> float:
    node = metrics
    for part in dotted.split("."):
        node = node[part]
    return float(node)


def _deployed_version(models_root: Path, model: str) -> str | None:
    prod = models_root / model / "PRODUCTION"
    return prod.read_text().strip() if prod.is_file() else None


_SEMVER = re.compile(r"^\d+\.\d+\.\d+$")


def _version_key(version: str) -> tuple[int, int, int]:
    return tuple(int(p) for p in version.split("."))


def _existing_versions(models_root: Path, model: str) -> list[str]:
    """All registered version directories for a model (semver-named only)."""
    model_dir = models_root / model
    if not model_dir.is_dir():
        return []
    return sorted((d.name for d in model_dir.iterdir()
                   if d.is_dir() and _SEMVER.match(d.name)),
                  key=_version_key)


def evaluate_gate(models_root: Path, model: str, candidate_version: str,
                  gate_metric: str = "test.auroc",
                  min_delta: float = 0.001) -> tuple[bool, dict]:
    """Evaluation gate: candidate must beat the incumbent by >= min_delta.

    Fail-closed rules:
      * The first-promotion shortcut applies ONLY when zero versions exist.
      * If versions exist but the PRODUCTION pointer is missing (or dangles),
        the candidate is compared against the LATEST existing version — the
        comparison is never skipped (ML-1: a missing pointer used to silently
        promote a same-seed duplicate).
      * If the incumbent's metrics cannot be read, the candidate is REJECTED.

    Returns (promoted, gate_record). The gate record always carries the
    candidate metric and, whenever an incumbent exists, the incumbent's
    version AND metric, so every notification is self-auditing.
    """
    models_root = Path(models_root)
    cand_metrics = json.loads(
        (models_root / model / candidate_version / "metrics.json").read_text())
    cand = _get_metric(cand_metrics, gate_metric)
    base = {"candidate_metric": cand, "min_delta": min_delta}

    deployed = _deployed_version(models_root, model)
    prior_versions = [v for v in _existing_versions(models_root, model)
                      if v != candidate_version]

    if deployed is None and not prior_versions:
        # genuine first-ever model: no incumbent exists to beat
        return True, {"incumbent": None, "incumbent_metric": None, **base,
                      "decision": "promote_first_model"}

    if deployed is not None and (models_root / model / deployed).is_dir():
        incumbent, source = deployed, "production_pointer"
    elif prior_versions:
        incumbent = max(prior_versions, key=_version_key)
        source = ("latest_version_no_pointer" if deployed is None
                  else "latest_version_pointer_invalid")
    else:
        # pointer exists but its target is gone and no other versions remain
        return False, {"incumbent": deployed, "incumbent_metric": None, **base,
                       "decision": "reject",
                       "reason": "production_pointer_invalid"}

    try:
        inc_metrics = json.loads(
            (models_root / model / incumbent / "metrics.json").read_text())
        inc = _get_metric(inc_metrics, gate_metric)
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        return False, {"incumbent": incumbent, "incumbent_source": source,
                       "incumbent_metric": None, **base, "decision": "reject",
                       "reason": f"incumbent_metrics_unreadable: {exc}"}

    promoted = cand >= inc + min_delta
    return promoted, {"incumbent": incumbent, "incumbent_source": source,
                      "incumbent_metric": inc, **base,
                      "decision": "promote" if promoted else "reject"}


def transition_stage(models_root: Path, model: str, version: str,
                     stage: str) -> None:
    """MLflow-registry-style stage transition, local implementation.

    With an MLflow server running, use `mlflow.register_model` +
    `transition_model_version_stage`; the local pointer files here mirror the
    same Staging/Production semantics so the gate logic is testable offline.
    """
    (models_root / model / stage).write_text(version + "\n")


def notify(event: dict, outbox: Path = Path("results/notifications.jsonl")) -> None:
    outbox.parent.mkdir(parents=True, exist_ok=True)
    event["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(outbox, "a") as f:
        f.write(json.dumps(event) + "\n")


def run_cycle(model: str, data_dir: str, candidate_version: str,
              models_root: str = "models", min_delta: float = 0.001,
              device: str = "cpu") -> dict:
    spec = MODEL_SPECS[model]
    models_root = Path(models_root)

    # 1. retrain candidate on the (possibly refreshed) snapshot
    cmd = [sys.executable, "-m", spec["train_module"], "--version", candidate_version,
           "--out", str(models_root / model), "--device", device]
    if spec["data_file"]:
        cmd += [spec["data_arg"], str(Path(data_dir) / spec["data_file"])]
    else:
        cmd += [spec["data_arg"], data_dir]
    subprocess.run(cmd, check=True)

    # 2. evaluation gate: candidate must beat incumbent on held-out metric
    promoted, gate = evaluate_gate(models_root, model, candidate_version,
                                   gate_metric=spec["gate_metric"],
                                   min_delta=min_delta)

    # 3. stage transition + 4. ONNX export + 5. notify
    transition_stage(models_root, model, candidate_version, "Staging")
    if promoted:
        subprocess.run([sys.executable, "-m", "inference.export_onnx",
                        "--model", model, "--version", candidate_version,
                        "--models-root", str(models_root),
                        "--data-dir", data_dir], check=True)
        transition_stage(models_root, model, candidate_version, "PRODUCTION")
        # keep PRODUCTION pointer and clean Staging pointer
        (models_root / model / "Staging").unlink(missing_ok=True)
    result = {"model": model, "candidate_version": candidate_version,
              "gate": gate, "promoted": promoted}
    notify({"event": "continuous_training_cycle", **result})
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(MODEL_SPECS))
    p.add_argument("--data-dir", default="data/synthetic")
    p.add_argument("--candidate-version", required=True)
    p.add_argument("--models-root", default="models")
    p.add_argument("--min-delta", type=float, default=0.001)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = p.parse_args()
    run_cycle(args.model, args.data_dir, args.candidate_version,
              models_root=args.models_root, min_delta=args.min_delta,
              device=args.device)


if __name__ == "__main__":
    main()
