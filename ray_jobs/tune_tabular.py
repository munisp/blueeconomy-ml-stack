"""Distributed hyperparameter search for the tabular fraud model (Ray Tune).

With a live Ray cluster this runs Ray Tune (ASHA scheduler); without Ray it
runs the identical search space as a sequential local sweep. Both paths report
the same best-config/best-metric and never invent results.

Usage:
    python -m ray_jobs.tune_tabular --data data/synthetic/declarations.parquet \
        --samples 8 [--address auto]
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ray_jobs.common import init_ray, _HAS_RAY
from synthetic.declarations import to_features
from training.common import classification_metrics, seed_everything
from training.tabular import FraudMLP

SEARCH_SPACE = {
    "hidden": [16, 32, 64],
    "lr": [1e-3, 3e-3],
    "dropout": [0.1, 0.2],
}


def _train_eval(config: dict, data_path: str, seed: int = 7, epochs: int = 40) -> dict:
    """One train/eval trial — identical under Ray and local fallback."""
    seed_everything(seed)
    df = pd.read_parquet(data_path)
    x, y = to_features(df)
    x_tr, x_val, y_tr, y_val = train_test_split(
        x, y, test_size=0.25, stratify=y, random_state=seed)
    model = FraudMLP(x.shape[1], hidden=config["hidden"], dropout=config["dropout"])
    pos_weight = torch.tensor([(y_tr == 0).sum() / max(1, (y_tr == 1).sum())])
    opt = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    xt = torch.tensor(x_tr); yt = torch.tensor(y_tr, dtype=torch.float32)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 4096):
            idx = perm[i:i + 4096]
            opt.zero_grad()
            loss = loss_fn(model(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(torch.tensor(x_val))).numpy()
    return {"config": config, "val_auroc": classification_metrics(y_val, scores)["auroc"]}


def _local_sweep(data_path: str, samples: int) -> list[dict]:
    combos = [dict(zip(SEARCH_SPACE, v)) for v in itertools.product(*SEARCH_SPACE.values())]
    rng = np.random.default_rng(7)
    order = rng.permutation(len(combos))[:samples]
    results = []
    for i in order:
        r = _train_eval(combos[i], data_path)
        print(f"[local-sweep] {r['config']} -> val_auroc={r['val_auroc']:.4f}")
        results.append(r)
    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/synthetic/declarations.parquet")
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--address", default="auto")
    p.add_argument("--out", default="results/tune_tabular.json")
    args = p.parse_args()

    live = init_ray(args.address, fallback_ok=True)
    if live:
        from ray import tune
        from ray.tune.schedulers import ASHAScheduler

        def trainable(config):
            r = _train_eval(config, args.data)
            tune.report({"val_auroc": r["val_auroc"]})

        tuner = tune.Tuner(
            tune.with_resources(trainable, {"cpu": 1}),
            param_space={k: tune.grid_search(v) if k == "hidden" else tune.choice(v)
                         for k, v in SEARCH_SPACE.items()},
            tune_config=tune.TuneConfig(num_samples=max(1, args.samples // 3),
                                        scheduler=ASHAScheduler(metric="val_auroc",
                                                                mode="max"),
                                        metric="val_auroc", mode="max"),
        )
        grid = tuner.fit()
        best = grid.get_best_result()
        summary = {"mode": "ray-tune", "best_config": best.config,
                   "best_val_auroc": float(best.metrics["val_auroc"])}
    else:
        results = _local_sweep(args.data, args.samples)
        best = max(results, key=lambda r: r["val_auroc"])
        summary = {"mode": "local-sweep (RAY_UNAVAILABLE fallback)",
                   "best_config": best["config"],
                   "best_val_auroc": best["val_auroc"],
                   "trials": results}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "trials"}, indent=2))


if __name__ == "__main__":
    main()
