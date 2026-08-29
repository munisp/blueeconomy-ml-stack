"""GNN trainer: shell-company / mule-account node classification.

Backend: PyTorch Geometric (SAGEConv) when installed; otherwise a pure-torch
GraphSAGE with identical mean-aggregation message passing semantics. Both are
REAL message passing over the trade/payment graph — no hashing tricks.

Usage:
    python -m training.gnn --data-dir data/synthetic \
        --out models/graph-mule-gnn --version 0.1.0 [--device cpu]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.common import (EarlyStopping, RunTracker, classification_metrics,
                             get_device, seed_everything)
from training.graph_data import build_graph

MODEL_NAME = "graph-mule-gnn"

try:
    from torch_geometric.nn import SAGEConv
    _HAS_PYG = True
except ImportError:  # pragma: no cover
    _HAS_PYG = False


class _TorchSAGEConv(nn.Module):
    """Pure-torch GraphSAGE layer (mean aggregation), PyG-semantics compatible."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_neigh = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        agg = torch.zeros_like(x).index_add_(0, dst, x[src])
        deg = torch.zeros(x.size(0), device=x.device).index_add_(
            0, dst, torch.ones(dst.size(0), device=x.device)).clamp_min_(1.0)
        agg = agg / deg.unsqueeze(-1)
        return self.lin_self(x) + self.lin_neigh(agg)


class GraphSAGE(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 24, n_layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        self.backend = "pyg" if _HAS_PYG else "torch-fallback"
        conv = SAGEConv if _HAS_PYG else _TorchSAGEConv
        dims = [in_dim] + [hidden] * n_layers
        self.convs = nn.ModuleList([conv(dims[i], dims[i + 1]) for i in range(n_layers)])
        self.head = nn.Linear(hidden, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = torch.relu(conv(x, edge_index))
            x = self.dropout(x)
        return self.head(x).squeeze(-1)


def train(args: argparse.Namespace) -> dict:
    seed_everything(args.seed)
    device = get_device(args.device)
    g = build_graph(args.data_dir, max_declarations=args.max_declarations, seed=args.seed)

    x = torch.tensor(g["x"]).to(device)
    edge_index = torch.tensor(g["edge_index"]).to(device)
    y = torch.tensor(g["y"], dtype=torch.float32).to(device)
    n_co = g["n_company"]

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_co)
    n_tr, n_val = int(0.6 * n_co), int(0.2 * n_co)
    idx_tr = torch.tensor(perm[:n_tr]).to(device)
    idx_val = torch.tensor(perm[n_tr:n_tr + n_val]).to(device)
    idx_te = torch.tensor(perm[n_tr + n_val:]).to(device)

    out_dir = Path(args.out) / args.version
    out_dir.mkdir(parents=True, exist_ok=True)

    with RunTracker("graph-mule", f"{MODEL_NAME}-{args.version}") as track:
        track.log_params({"model": MODEL_NAME, "version": args.version,
                          "backend": "pyg" if _HAS_PYG else "torch-fallback",
                          "seed": args.seed, "device": str(device),
                          **{f"graph_{k}": v for k, v in g["meta"].items()}})

        model = GraphSAGE(g["node_feature_dim"], hidden=args.hidden).to(device)
        pos_weight = (y[idx_tr] == 0).sum() / (y[idx_tr] == 1).sum().clamp_min(1)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-4)
        stopper = EarlyStopping(patience=15)
        best_state, best_val = None, -1.0

        for epoch in range(args.epochs):
            model.train()
            opt.zero_grad()
            logits = model(x, edge_index)
            loss = loss_fn(logits[idx_tr], y[idx_tr])
            loss.backward()
            opt.step()
            model.eval()
            with torch.no_grad():
                val = torch.sigmoid(model(x, edge_index)[idx_val]).cpu().numpy()
            m = classification_metrics(y[idx_val].cpu().numpy().astype(int), val)
            track.log_metrics({"train_loss": float(loss), "val_auroc": m["auroc"]}, step=epoch)
            if m["auroc"] > best_val:
                best_val = m["auroc"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if stopper.step(m["auroc"]):
                print(f"[early-stop] epoch={epoch} best_val_auroc={best_val:.4f}")
                break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            te = torch.sigmoid(model(x, edge_index)[idx_te]).cpu().numpy()
        metrics = classification_metrics(y[idx_te].cpu().numpy().astype(int), te)
        track.log_metrics({f"test_{k}": v for k, v in metrics.items()})

        torch.save({"state_dict": model.state_dict(),
                    "in_dim": g["node_feature_dim"], "hidden": args.hidden,
                    "backend": model.backend}, out_dir / "model.pt")
        payload = {"model": MODEL_NAME, "version": args.version,
                   "backend": model.backend, "data_source": "SYNTHETIC",
                   "graph": g["meta"], "test": metrics}
        (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
        track.log_artifact(out_dir / "metrics.json")

    print(f"[{MODEL_NAME}] backend={model.backend} test: " +
          ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    return payload


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/synthetic")
    p.add_argument("--out", default="models/graph-mule-gnn")
    p.add_argument("--version", required=True)
    p.add_argument("--hidden", type=int, default=24)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--max-declarations", type=int, default=4000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    return p


if __name__ == "__main__":
    train(build_argparser().parse_args())
