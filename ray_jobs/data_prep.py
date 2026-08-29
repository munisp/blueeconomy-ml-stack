"""Distributed data prep with Ray Data (local fallback: chunked pandas).

Reads raw parquet shards, applies the feature transforms from the synthetic
package, and writes feature-table parquet shards. With a live Ray cluster the
map is distributed; otherwise it runs sequentially with identical output.

Usage:
    python -m ray_jobs.data_prep --in data/synthetic --out data/features
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ray_jobs.common import init_ray
from synthetic.ais import AIS_FEATURES
from synthetic.declarations import FEATURE_COLUMNS, to_features as decl_features
from synthetic.ais import to_features as ais_features


def _decl_batch(batch: pd.DataFrame) -> pd.DataFrame:
    x, y = decl_features(batch)
    out = pd.DataFrame(x, columns=FEATURE_COLUMNS)
    out["label"] = y
    return out


def _ais_batch(batch: pd.DataFrame) -> pd.DataFrame:
    x, y = ais_features(batch)
    out = pd.DataFrame(x, columns=AIS_FEATURES)
    out["label"] = y
    return out


def run(data_in: str, out_dir: str, address: str = "auto") -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    live = init_ray(address, fallback_ok=True)
    written = []

    jobs = [("declarations", _decl_batch), ("ais", _ais_batch)]
    if live:
        import ray
        for name, fn in jobs:
            ds = ray.data.read_parquet(str(Path(data_in) / f"{name}.parquet"))
            ds.map_batches(fn, batch_format="pandas").write_parquet(
                str(out / name))
            written.append(name)
        mode = "ray-data"
    else:
        for name, fn in jobs:
            df = pd.read_parquet(Path(data_in) / f"{name}.parquet")
            fn(df).to_parquet(out / f"{name}.parquet", index=False)
            written.append(name)
        mode = "local (RAY_UNAVAILABLE fallback)"
    summary = {"mode": mode, "feature_tables": written, "out": str(out)}
    print(summary)
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="data_in", default="data/synthetic")
    p.add_argument("--out", default="data/features")
    p.add_argument("--address", default="auto")
    args = p.parse_args()
    run(args.data_in, args.out, args.address)


if __name__ == "__main__":
    main()
