"""Export trained PyTorch checkpoints to ONNX (opset 17) for CPU inference.

Exports the tabular MLP and the vessel autoencoder. The GNN requires a graph
edge_index input; it is exported with a fixed-graph wrapper when its training
graph is supplied, otherwise served via the torch checkpoint (documented).

Usage:
    python -m inference.export_onnx --model declaration-fraud --version 0.1.0
    python -m inference.export_onnx --model vessel-anomaly --version 0.1.0
    python -m inference.export_onnx --model graph-mule-gnn --version 0.1.0 \
        --data-dir data/synthetic
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _verify(onnx_path: Path, sample: np.ndarray, reference: float) -> None:
    """Load in onnxruntime and check output parity vs the torch reference."""
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    out = float(sess.run(None, {name: sample})[0].reshape(-1)[0])
    if abs(out - reference) > 1e-3:
        raise RuntimeError(f"ONNX parity check failed: onnx={out} torch={reference}")
    size_kb = onnx_path.stat().st_size / 1024
    if size_kb > 5 * 1024:
        raise RuntimeError(f"ONNX artifact {size_kb:.0f}KB exceeds 5MB budget")
    print(f"[verify] {onnx_path} parity ok ({size_kb:.0f}KB)")


def _stamp_kind(ver_dir: Path, kind: str) -> None:
    meta = json.loads((ver_dir / "metrics.json").read_text())
    meta["kind"] = kind
    (ver_dir / "metrics.json").write_text(json.dumps(meta, indent=2))


def export_tabular(models_root: Path, version: str) -> None:
    from training.tabular import FraudMLP
    ver_dir = models_root / "declaration-fraud" / version
    ckpt = torch.load(ver_dir / "model.pt", map_location="cpu", weights_only=False)
    model = FraudMLP(ckpt["n_features"], hidden=ckpt["hidden"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    class _Calibrated(torch.nn.Module):  # export calibrated logits
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, features):
            return self.m.calibrated_logits(features)

    wrapped = _Calibrated(model)
    sample = np.zeros((1, ckpt["n_features"]), dtype=np.float32)
    with torch.no_grad():
        ref = float(wrapped(torch.tensor(sample)))
    out = ver_dir / "model.onnx"
    torch.onnx.export(wrapped, (torch.tensor(sample),), str(out), opset_version=17,
                      input_names=["features"], output_names=["logit"],
                      dynamic_axes={"features": {0: "batch"}})
    _verify(out, sample, ref)
    _stamp_kind(ver_dir, "classifier")


def export_autoencoder(models_root: Path, version: str) -> None:
    from training.anomaly import Autoencoder
    ver_dir = models_root / "vessel-anomaly" / version
    ckpt = torch.load(ver_dir / "model.pt", map_location="cpu", weights_only=False)
    model = Autoencoder(ckpt["n_features"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    class _Score(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, features):
            return self.m.anomaly_score(features)

    wrapped = _Score(model)
    sample = np.zeros((1, ckpt["n_features"]), dtype=np.float32)
    with torch.no_grad():
        ref = float(wrapped(torch.tensor(sample)))
    out = ver_dir / "model.onnx"
    torch.onnx.export(wrapped, (torch.tensor(sample),), str(out), opset_version=17,
                      input_names=["features"], output_names=["anomaly_score"],
                      dynamic_axes={"features": {0: "batch"}})
    _verify(out, sample, ref)
    _stamp_kind(ver_dir, "autoencoder")


def export_gnn(models_root: Path, version: str, data_dir: str) -> None:
    """Export GraphSAGE with a fixed training graph baked in (transductive).

    ONNX input is the company-node feature slice only; the graph adjacency is
    frozen at export time. Inductive (new-graph) scoring must use the .pt
    checkpoint — this limitation is documented in the model card.
    """
    from training.gnn import GraphSAGE
    from training.graph_data import build_graph
    ver_dir = models_root / "graph-mule-gnn" / version
    ckpt = torch.load(ver_dir / "model.pt", map_location="cpu", weights_only=False)
    model = GraphSAGE(ckpt["in_dim"], hidden=ckpt["hidden"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    g = build_graph(data_dir)
    edge_index = torch.tensor(g["edge_index"])
    n_co, in_dim = g["n_company"], g["node_feature_dim"]
    pad = g["x"].shape[0] - n_co  # non-company context nodes frozen at export

    class _FixedGraph(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
            self.register_buffer("ctx", torch.zeros(pad, in_dim))
        def forward(self, features):
            x = torch.cat([features, self.ctx.to(features.dtype)], dim=0)
            return self.m(x, edge_index)[:features.shape[0]]

    wrapped = _FixedGraph(model)
    with torch.no_grad():
        wrapped.ctx.copy_(torch.tensor(g["x"][n_co:]))
    sample = torch.tensor(g["x"][:n_co])
    with torch.no_grad():
        ref = float(wrapped(sample)[0])
    out = ver_dir / "model.onnx"
    torch.onnx.export(wrapped, (sample,), str(out), opset_version=17,
                      input_names=["features"], output_names=["logits"],
                      dynamic_axes={"features": {0: "n_company"}})
    _verify(out, sample.numpy(), ref)
    _stamp_kind(ver_dir, "classifier")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   choices=["declaration-fraud", "vessel-anomaly", "graph-mule-gnn"])
    p.add_argument("--version", required=True)
    p.add_argument("--models-root", default="models")
    p.add_argument("--data-dir", default="data/synthetic")
    args = p.parse_args()
    root = Path(args.models_root)
    if args.model == "declaration-fraud":
        export_tabular(root, args.version)
    elif args.model == "vessel-anomaly":
        export_autoencoder(root, args.version)
    else:
        export_gnn(root, args.version, args.data_dir)


if __name__ == "__main__":
    main()
