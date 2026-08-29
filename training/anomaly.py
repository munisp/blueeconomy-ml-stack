"""Anomaly detection trainer: Isolation Forest baseline + PyTorch autoencoder.

Trains on vessel AIS movement features (normal pings only for the AE, the
honest semi-supervised setup), scores anomalies by reconstruction error, and
benchmarks against an Isolation Forest on identical splits.

Usage:
    python -m training.anomaly --data data/synthetic/ais.parquet \
        --out models/vessel-anomaly --version 0.1.0 [--device cpu]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from synthetic.ais import AIS_FEATURES, to_features
from training.common import (EarlyStopping, RunTracker, classification_metrics,
                             get_device, seed_everything)

MODEL_NAME = "vessel-anomaly-autoencoder"
BASELINE_NAME = "vessel-anomaly-isoforest"


class Autoencoder(nn.Module):
    def __init__(self, n_features: int, latent: int = 4, hidden: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(), nn.Linear(hidden, latent))
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, n_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        return ((self.forward(x) - x) ** 2).mean(dim=1)


def train(args: argparse.Namespace) -> dict:
    seed_everything(args.seed)
    device = get_device(args.device)
    df = pd.read_parquet(args.data)
    x, y = to_features(df)

    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=0.30, stratify=y, random_state=args.seed)
    x_tr, x_val, y_tr, y_val = train_test_split(
        x_tr, y_tr, test_size=0.20, stratify=y_tr, random_state=args.seed)
    x_tr_normal = x_tr[y_tr == 0]  # semi-supervised: train AE on normal only

    out_dir = Path(args.out) / args.version
    out_dir.mkdir(parents=True, exist_ok=True)

    with RunTracker("vessel-anomaly", f"{MODEL_NAME}-{args.version}") as track:
        track.log_params({"model": MODEL_NAME, "version": args.version,
                          "seed": args.seed, "device": str(device),
                          "features": ",".join(AIS_FEATURES),
                          "n_train_normal": len(x_tr_normal),
                          "data_source": df["data_source"].iloc[0]})

        ae = Autoencoder(x.shape[1]).to(device)
        opt = torch.optim.AdamW(ae.parameters(), lr=3e-3)
        loss_fn = nn.MSELoss()
        stopper = EarlyStopping(patience=12)
        xt = torch.tensor(x_tr_normal).to(device)
        xv = torch.tensor(x_val).to(device)
        best_state, best_val = None, -1.0

        for epoch in range(args.epochs):
            ae.train()
            perm = torch.randperm(len(xt))
            for i in range(0, len(xt), 4096):
                idx = perm[i:i + 4096]
                opt.zero_grad()
                loss = loss_fn(ae(xt[idx]), xt[idx])
                loss.backward()
                opt.step()
            ae.eval()
            with torch.no_grad():
                val_scores = ae.anomaly_score(xv).cpu().numpy()
            m = classification_metrics(y_val, val_scores)
            track.log_metrics({"train_loss": float(loss), "val_auroc": m["auroc"]}, step=epoch)
            if m["auroc"] > best_val:
                best_val = m["auroc"]
                best_state = {k: v.clone() for k, v in ae.state_dict().items()}
            if stopper.step(m["auroc"]):
                print(f"[early-stop] epoch={epoch} best_val_auroc={best_val:.4f}")
                break

        ae.load_state_dict(best_state)
        ae.eval()
        with torch.no_grad():
            te_scores = ae.anomaly_score(torch.tensor(x_te).to(device)).cpu().numpy()
        ae_metrics = classification_metrics(y_te, te_scores)
        track.log_metrics({f"test_{k}": v for k, v in ae_metrics.items()})

        # --- Isolation Forest baseline ---
        iso = IsolationForest(n_estimators=200, contamination=float(y_tr.mean()),
                              random_state=args.seed, n_jobs=-1)
        iso.fit(x_tr_normal)
        iso_metrics = classification_metrics(y_te, -iso.score_samples(x_te))
        track.log_metrics({f"baseline_test_{k}": v for k, v in iso_metrics.items()})

        torch.save({"state_dict": ae.state_dict(), "n_features": x.shape[1],
                    "feature_columns": AIS_FEATURES}, out_dir / "model.pt")
        import joblib
        joblib.dump(iso, out_dir / "baseline_isoforest.joblib")
        payload = {"model": MODEL_NAME, "version": args.version,
                   "data_source": "SYNTHETIC", "test": ae_metrics,
                   "baseline_isoforest_test": iso_metrics}
        (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
        track.log_artifact(out_dir / "metrics.json")

    print(f"[{MODEL_NAME}] test: " + ", ".join(f"{k}={v:.4f}" for k, v in ae_metrics.items()))
    print(f"[{BASELINE_NAME}] test: " + ", ".join(f"{k}={v:.4f}" for k, v in iso_metrics.items()))
    return payload


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/synthetic/ais.parquet")
    p.add_argument("--out", default="models/vessel-anomaly")
    p.add_argument("--version", required=True)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    return p


if __name__ == "__main__":
    train(build_argparser().parse_args())
