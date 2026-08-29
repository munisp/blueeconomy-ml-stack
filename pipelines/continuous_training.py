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
import shutil
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

    # 2. evaluation gate: candidate must beat deployed on held-out metric
    cand_metrics = json.loads(
        (models_root / model / candidate_version / "metrics.json").read_text())
    cand = _get_metric(cand_metrics, spec["gate_metric"])
    deployed_version = _deployed_version(models_root, model)
    promoted = False
    if deployed_version is None:
        promoted = True  # first model: promote (no incumbent to beat)
        gate = {"incumbent": None, "decision": "promote_first_model"}
    else:
        inc_metrics = json.loads(
            (models_root / model / deployed_version / "metrics.json").read_text())
        inc = _get_metric(inc_metrics, spec["gate_metric"])
        promoted = cand >= inc + min_delta
        gate = {"incumbent": deployed_version, "incumbent_metric": inc,
                "candidate_metric": cand, "min_delta": min_delta,
                "decision": "promote" if promoted else "reject"}

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
