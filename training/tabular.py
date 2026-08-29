"""Tabular customs fraud/undervaluation trainer.

Trains (1) a small PyTorch MLP with class-imbalance handling + temperature
calibration and (2) a LightGBM gradient-boosting baseline for honest
comparison. Both are evaluated on the same held-out test split and all
metrics are logged. Weights are kept tiny by design (<5MB).

Usage:
    python -m training.tabular --data data/synthetic/declarations.parquet \
        --out models/declaration-fraud --version 0.1.0 [--device cpu]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from synthetic.declarations import FEATURE_COLUMNS, to_features
from training.common import (EarlyStopping, RunTracker, classification_metrics,
                             get_device, seed_everything)

MODEL_NAME = "declaration-fraud-mlp"
BASELINE_NAME = "declaration-fraud-lightgbm"


class FraudMLP(nn.Module):
    """Deliberately small MLP — CPU inference budget and <5MB artifact."""

    def __init__(self, n_features: int, hidden: int = 32, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.temperature = nn.Parameter(torch.ones(1))  # post-hoc calibration

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def calibrated_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x) / self.temperature.clamp_min(1e-3)


def _fit_temperature(model: FraudMLP, x_val: torch.Tensor, y_val: torch.Tensor) -> None:
    opt = torch.optim.LBFGS([model.temperature], max_iter=50)
    loss_fn = nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        loss = loss_fn(model.calibrated_logits(x_val), y_val)
        loss.backward()
        return loss
    opt.step(closure)


def train(args: argparse.Namespace) -> dict:
    seed_everything(args.seed)
    device = get_device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.data)
    x, y = to_features(df)
    x_tr, x_tmp, y_tr, y_tmp = train_test_split(
        x, y, test_size=0.30, stratify=y, random_state=args.seed)
    x_val, x_te, y_val, y_te = train_test_split(
        x_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=args.seed)

    pos_weight = torch.tensor([(y_tr == 0).sum() / max(1, (y_tr == 1).sum())])

    with RunTracker("declaration-fraud", f"{MODEL_NAME}-{args.version}") as track:
        track.log_params({
            "model": MODEL_NAME, "version": args.version, "seed": args.seed,
            "n_train": len(x_tr), "n_val": len(x_val), "n_test": len(x_te),
            "features": ",".join(FEATURE_COLUMNS), "device": str(device),
            "hidden": args.hidden, "epochs_max": args.epochs,
            "data_source": df["data_source"].iloc[0],
        })

        model = FraudMLP(x.shape[1], hidden=args.hidden).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
        stopper = EarlyStopping(patience=10)
        xt = torch.tensor(x_tr).to(device); yt = torch.tensor(y_tr, dtype=torch.float32).to(device)
        xv = torch.tensor(x_val).to(device); yv = torch.tensor(y_val, dtype=torch.float32).to(device)

        best_state, best_val = None, -1.0
        for epoch in range(args.epochs):
            model.train()
            perm = torch.randperm(len(xt))
            total = 0.0
            for i in range(0, len(xt), 2048):
                idx = perm[i:i + 2048]
                opt.zero_grad()
                loss = loss_fn(model(xt[idx]), yt[idx])
                loss.backward()
                opt.step()
                total += float(loss) * len(idx)
            model.eval()
            with torch.no_grad():
                val_scores = torch.sigmoid(model(xv)).cpu().numpy()
            val_auroc = classification_metrics(y_val, val_scores)["auroc"]
            track.log_metrics({"train_loss": total / len(xt), "val_auroc": val_auroc}, step=epoch)
            if val_auroc > best_val:
                best_val = val_auroc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if stopper.step(val_auroc):
                print(f"[early-stop] epoch={epoch} best_val_auroc={best_val:.4f}")
                break

        model.load_state_dict(best_state)
        with torch.no_grad():
            _fit_temperature(model, xv, yv)
            te_scores = torch.sigmoid(model.calibrated_logits(
                torch.tensor(x_te).to(device))).cpu().numpy()
        mlp_metrics = classification_metrics(y_te, te_scores)
        mlp_metrics["temperature"] = float(model.temperature)
        track.log_metrics({f"test_{k}": v for k, v in mlp_metrics.items()})

        # --- LightGBM baseline for honest comparison ---
        import lightgbm as lgb
        lgbm = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            scale_pos_weight=float(pos_weight), random_state=args.seed, verbose=-1)
        lgbm.fit(x_tr, y_tr, eval_set=[(x_val, y_val)],
                 callbacks=[lgb.early_stopping(30, verbose=False)])
        lgbm_cal = CalibratedClassifierCV(lgbm, method="isotonic", cv="prefit")
        lgbm_cal.fit(x_val, y_val)
        lgbm_metrics = classification_metrics(y_te, lgbm_cal.predict_proba(x_te)[:, 1])
        track.log_metrics({f"baseline_test_{k}": v for k, v in lgbm_metrics.items()})

        # --- persist versioned artifacts ---
        ver_dir = out_dir / args.version
        ver_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(),
                    "n_features": x.shape[1], "hidden": args.hidden,
                    "feature_columns": FEATURE_COLUMNS}, ver_dir / "model.pt")
        import joblib
        joblib.dump(lgbm_cal, ver_dir / "baseline_lightgbm.joblib")
        metrics = {
            "model": MODEL_NAME, "version": args.version,
            "data_source": "SYNTHETIC",
            "test": mlp_metrics,
            "baseline_lightgbm_test": lgbm_metrics,
            "splits": {"train": len(x_tr), "val": len(x_val), "test": len(x_te)},
        }
        (ver_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        track.log_artifact(ver_dir / "metrics.json")

    print(f"[{MODEL_NAME}] test: " +
          ", ".join(f"{k}={v:.4f}" for k, v in mlp_metrics.items() if isinstance(v, float)))
    print(f"[{BASELINE_NAME}] test: " +
          ", ".join(f"{k}={v:.4f}" for k, v in lgbm_metrics.items()))
    return metrics


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/synthetic/declarations.parquet")
    p.add_argument("--out", default="models/declaration-fraud")
    p.add_argument("--version", required=True)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    return p


if __name__ == "__main__":
    train(build_argparser().parse_args())
