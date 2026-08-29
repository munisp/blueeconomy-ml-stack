"""CLI: generate all SYNTHETIC datasets to Parquet with a version manifest.

Usage:
    python -m synthetic.cli --out data/synthetic [--declarations 20000] ...

Writes:
    <out>/declarations.parquet      SYNTHETIC customs declarations
    <out>/ais.parquet               SYNTHETIC vessel AIS pings
    <out>/cvff_companies.parquet    SYNTHETIC company nodes (GNN labels)
    <out>/cvff_accounts.parquet     SYNTHETIC account nodes
    <out>/cvff_transactions.parquet SYNTHETIC CVFF/payment transactions
    <out>/manifest.json             dataset_version hash + config + row counts
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from .ais import generate_ais
from .config import AisConfig, CvffConfig, DeclarationConfig, SyntheticConfig
from .cvff import generate_cvff
from .declarations import generate_declarations


def dataset_version(df: pd.DataFrame) -> str:
    """Deterministic content hash of a dataframe (dataset_version)."""
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes())
    h.update("|".join(df.columns).encode())
    return h.hexdigest()[:16]


def main() -> None:
    p = argparse.ArgumentParser(description="Generate SYNTHETIC BlueEconomy training data")
    p.add_argument("--out", default="data/synthetic")
    p.add_argument("--declarations", type=int, default=20_000)
    p.add_argument("--vessels", type=int, default=120)
    p.add_argument("--days", type=int, default=21)
    p.add_argument("--companies", type=int, default=1_800)
    p.add_argument("--transactions", type=int, default=40_000)
    p.add_argument("--seed", type=int, default=20240)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = SyntheticConfig(
        declarations=DeclarationConfig(n_declarations=args.declarations, seed=args.seed),
        ais=AisConfig(n_vessels=args.vessels, days=args.days, seed=args.seed + 1),
        cvff=CvffConfig(n_companies=args.companies, n_transactions=args.transactions,
                        seed=args.seed + 2),
    )

    decl = generate_declarations(cfg.declarations)
    ais = generate_ais(cfg.ais)
    cvff = generate_cvff(cfg.cvff)

    frames = {
        "declarations": decl,
        "ais": ais,
        "cvff_companies": cvff["companies"],
        "cvff_accounts": cvff["accounts"],
        "cvff_transactions": cvff["transactions"],
    }
    manifest = {"data_source": "SYNTHETIC", "config": json.loads(cfg.to_json()),
                "datasets": {}}
    for name, df in frames.items():
        path = out / f"{name}.parquet"
        df.to_parquet(path, index=False)
        manifest["datasets"][name] = {
            "file": path.name, "rows": len(df),
            "dataset_version": dataset_version(df),
        }
        print(f"[SYNTHETIC] wrote {path} rows={len(df)} "
              f"version={manifest['datasets'][name]['dataset_version']}")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[SYNTHETIC] manifest -> {out / 'manifest.json'}")


if __name__ == "__main__":
    main()
