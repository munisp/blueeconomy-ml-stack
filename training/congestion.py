"""Port-congestion forecaster training path from REAL queue observations (G7).

Trains a small CPU-budget regression MLP predicting the port queue length
HORIZON_MINUTES ahead, from the recorded port_queue_observations table
(geo-service migration 0013). Data source is REAL platform observations
only — no synthetic rows are ever mixed in.

Config-gated and fail-closed:
- BEML_CONGESTION_PG_DSN unset           -> honest exit (training disabled)
- fewer than --min-samples supervised pairs -> honest exit
  (INSUFFICIENT_HISTORY; the registry keeps reporting SCORING_UNAVAILABLE
  and geo-service keeps answering 409 until a real artifact lands)
- the exported artifact is models/port-congestion/<version>/{model.onnx,
  metrics.json}; metrics are computed on a held-out time-ordered tail, so
  the reported MAE is a genuinely out-of-sample number.

Usage:
    BEML_CONGESTION_PG_DSN=postgres://... \
    python -m training.congestion --out models/port-congestion --version 0.1.0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.common import EarlyStopping, RunTracker, get_device, seed_everything

MODEL_NAME = "port-congestion"
HORIZON_MINUTES = 60
# Feature vector (order is contractual — the serving caller must match):
FEATURES = [
    "queue_now",        # current queue length
    "queue_lag1",       # previous observation
    "queue_mean_3",     # mean of the last 3 observations
    "queue_delta1",     # queue_now - queue_lag1
    "minutes_since_prev",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "port_index_norm",  # stable per-port index / n_ports (baked at training)
]


def load_observations(dsn: str) -> pd.DataFrame:
    """Reads port_queue_observations via psycopg (imported lazily so the
    inference/test environments carry no DB driver)."""
    try:
        import psycopg
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise SystemExit(
            "BEML_CONGESTION: psycopg is required for training "
            "(pip install 'psycopg[binary]'); inference does not need it"
        ) from exc
    with psycopg.connect(dsn) as conn:
        rows = conn.execute(
            "SELECT port_code, queue_length, observed_at "
            "FROM port_queue_observations ORDER BY port_code, observed_at"
        ).fetchall()
    df = pd.DataFrame(rows, columns=["port_code", "queue_length", "observed_at"])
    df["observed_at"] = pd.to_datetime(df["observed_at"], utc=True)
    df["queue_length"] = df["queue_length"].astype(np.float32)
    df["data_source"] = "REAL:port_queue_observations"
    return df


def build_supervised(df: pd.DataFrame, horizon_minutes: int = HORIZON_MINUTES) -> tuple[np.ndarray, np.ndarray, dict]:
    """Builds (features, target) pairs: target is the first observation at
    least horizon_minutes after the anchor (tolerance window h..h*1.5);
    anchors without a real future observation are dropped, never imputed."""
    ports = sorted(df["port_code"].unique())
    port_index = {p: i / max(1, len(ports)) for i, p in enumerate(ports)}
    xs, ys = [], []
    for port, group in df.groupby("port_code"):
        g = group.sort_values("observed_at").reset_index(drop=True)
        times = g["observed_at"].to_numpy()
        qlen = g["queue_length"].to_numpy(dtype=np.float64)
        for i in range(len(g)):
            t0 = times[i]
            # horizon target: first observation in [t0+h, t0+1.5h)
            lo = t0 + np.timedelta64(horizon_minutes, "m")
            hi = t0 + np.timedelta64(int(horizon_minutes * 1.5), "m")
            future = np.searchsorted(times, lo)
            if future >= len(g) or times[future] >= hi or future == i:
                continue
            prev = qlen[i - 1] if i > 0 else qlen[i]
            prev_t = times[i - 1] if i > 0 else times[i]
            mean3 = qlen[max(0, i - 2): i + 1].mean()
            ts = pd.Timestamp(t0)
            hour = ts.hour + ts.minute / 60.0
            xs.append([
                qlen[i], prev, mean3, qlen[i] - prev,
                (t0 - prev_t) / np.timedelta64(1, "m"),
                np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
                np.sin(2 * np.pi * ts.dayofweek / 7), np.cos(2 * np.pi * ts.dayofweek / 7),
                port_index[port],
            ])
            ys.append(qlen[future])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32), port_index


class CongestionMLP(nn.Module):
    """Small regression MLP — CPU inference budget, <5MB artifact."""

    def __init__(self, n_features: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train(args: argparse.Namespace) -> dict:
    seed_everything(args.seed)
    device = get_device(args.device)
    dsn = os.environ.get("BEML_CONGESTION_PG_DSN", "").strip()
    if not dsn:
        raise SystemExit(
            "BEML_CONGESTION_PG_DSN is not set: congestion training is "
            "config-gated and disabled. The serving registry continues to "
            "report SCORING_UNAVAILABLE for port-congestion (fail-closed)."
        )
    df = load_observations(dsn)
    x, y, port_index = build_supervised(df, args.horizon_minutes)
    if len(x) < args.min_samples:
        raise SystemExit(
            f"INSUFFICIENT_HISTORY: {len(x)} supervised pairs from "
            f"port_queue_observations (< {args.min_samples}); no model "
            f"trained, registry stays SCORING_UNAVAILABLE."
        )
    # Time-ordered holdout: last 20% of pairs is the honest out-of-sample tail.
    split = int(len(x) * 0.8)
    x_tr, x_te = x[:split], x[split:]
    y_tr, y_te = y[:split], y[split:]

    with RunTracker(MODEL_NAME, f"{MODEL_NAME}-{args.version}") as track:
        track.log_params({
            "model": MODEL_NAME, "version": args.version, "seed": args.seed,
            "n_train": len(x_tr), "n_test": len(x_te),
            "horizon_minutes": args.horizon_minutes,
            "features": ",".join(FEATURES), "device": str(device),
            "data_source": df["data_source"].iloc[0],
            "ports": ",".join(sorted(df["port_code"].unique())),
        })
        model = CongestionMLP(x.shape[1], hidden=args.hidden).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
        loss_fn = nn.L1Loss()
        stopper = EarlyStopping(patience=10)
        xt = torch.tensor(x_tr).to(device)
        yt = torch.tensor(y_tr, dtype=torch.float32).to(device)
        best_state, best_val = None, np.inf
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
                val_mae = float(loss_fn(model(torch.tensor(x_te).to(device)),
                                        torch.tensor(y_te, dtype=torch.float32).to(device)))
            track.log_metrics({"train_mae": total / len(xt), "test_mae": val_mae}, step=epoch)
            if val_mae < best_val:
                best_val, best_state = val_mae, {k: v.clone() for k, v in model.state_dict().items()}
            if stopper.step(-val_mae):
                print(f"[early-stop] epoch={epoch} best_test_mae={best_val:.4f}")
                break
        model.load_state_dict(best_state)

        out_dir = Path(args.out) / args.version
        out_dir.mkdir(parents=True, exist_ok=True)
        model.eval()
        dummy = torch.zeros(1, x.shape[1])
        torch.onnx.export(
            model.cpu(), dummy, out_dir / "model.onnx",
            input_names=["features"], output_names=["score"],
            dynamic_axes={"features": {0: "batch"}, "score": {0: "batch"}},
            opset_version=17,
        )
        metrics = {
            "kind": "regressor", "model": MODEL_NAME, "version": args.version,
            "horizon_minutes": args.horizon_minutes,
            "test_mae_queue_length": best_val, "n_test": len(x_te),
            "features": FEATURES, "port_index": port_index,
            "data_source": "REAL:port_queue_observations",
        }
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        track.log_artifact(out_dir / "model.onnx")
        print(f"[done] {MODEL_NAME} {args.version}: test MAE {best_val:.3f} queue length "
              f"(n_test={len(x_te)}) -> {out_dir}")
        return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="models/port-congestion")
    parser.add_argument("--version", default="0.1.0")
    parser.add_argument("--horizon-minutes", type=int, default=HORIZON_MINUTES)
    parser.add_argument("--min-samples", type=int, default=500)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
