"""Training-data extraction from the platform lakehouse.

Mirrors blueeconomy-data-platform conventions:
- storage root resolved from environment only (no hardcoded endpoints);
  BEML_LAKEHOUSE_ROOT for dev, or BLUEECONOMY_* vars when colocated with the
  data platform deployment.
- medallion layout: bronze (raw events) -> silver (conformed) -> gold
  (feature/training marts). This pipeline consumes gold and emits versioned
  training snapshots with dataset_version content hashes.
- fail-closed: if the lakehouse root is unset or a dataset is absent, we do
  NOT silently fabricate production data — we either raise, or, when
  explicitly enabled, fall back to clearly-labelled SYNTHETIC data.

Incremental builder: production events -> feature tables -> training
snapshots. A snapshot is immutable and named by its dataset_version hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ENV_LAKEHOUSE_ROOT = "BEML_LAKEHOUSE_ROOT"
ENV_ALLOW_SYNTHETIC_FALLBACK = "BEML_ALLOW_SYNTHETIC_FALLBACK"

GOLD_DATASETS = {
    "declarations": "gold/customs/declarations.parquet",
    "ais": "gold/vessels/ais_positions.parquet",
    "cvff_companies": "gold/cvff/companies.parquet",
    "cvff_accounts": "gold/cvff/accounts.parquet",
    "cvff_transactions": "gold/cvff/transactions.parquet",
}


class LakehouseConfigurationError(RuntimeError):
    """Fail-closed: missing/invalid lakehouse configuration."""


@dataclass(frozen=True)
class ExtractionConfig:
    min_rows: int = 10_000          # below this -> synthetic fallback (if allowed)
    synthetic_seed: int = 20240
    snapshot_root: str = "data/snapshots"


def resolve_lakehouse_root() -> Path | None:
    root = os.environ.get(ENV_LAKEHOUSE_ROOT, "")
    if not root:
        return None
    p = Path(root)
    return p if p.is_dir() else None


def dataset_version(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes())
    h.update("|".join(map(str, df.columns)).encode())
    return h.hexdigest()[:16]


def _synthetic_fallback(name: str, seed: int) -> pd.DataFrame:
    """Clearly-labelled SYNTHETIC stand-in for a gold dataset."""
    from synthetic.ais import generate_ais
    from synthetic.config import AisConfig, CvffConfig, DeclarationConfig
    from synthetic.cvff import generate_cvff
    from synthetic.declarations import generate_declarations
    if name == "declarations":
        return generate_declarations(DeclarationConfig(seed=seed))
    if name == "ais":
        return generate_ais(AisConfig(seed=seed + 1))
    cvff = generate_cvff(CvffConfig(seed=seed + 2))
    key = name.replace("cvff_", "")
    if key in cvff:
        return cvff[key]
    raise LakehouseConfigurationError(f"no synthetic fallback for dataset '{name}'")


def extract_dataset(name: str, cfg: ExtractionConfig) -> tuple[pd.DataFrame, dict]:
    """Return (dataframe, lineage) for one dataset, fail-closed semantics."""
    lineage: dict = {"dataset": name}
    root = resolve_lakehouse_root()
    df = None
    if root is not None:
        path = root / GOLD_DATASETS[name]
        if path.is_file():
            df = pd.read_parquet(path)
            lineage.update(source="lakehouse", path=str(path))
    if df is None or len(df) < cfg.min_rows:
        if os.environ.get(ENV_ALLOW_SYNTHETIC_FALLBACK, "") != "1":
            raise LakehouseConfigurationError(
                f"dataset '{name}' unavailable or below min_rows={cfg.min_rows} and "
                f"{ENV_ALLOW_SYNTHETIC_FALLBACK}!=1; refusing to fabricate data")
        df = _synthetic_fallback(name, cfg.synthetic_seed)
        lineage.update(source="SYNTHETIC_FALLBACK",
                       reason="lakehouse absent or below volume threshold")
    lineage.update(rows=len(df), dataset_version=dataset_version(df),
                   data_source=str(df.get("data_source", pd.Series(["PRODUCTION"])).iloc[0]))
    return df, lineage


def build_training_snapshot(cfg: ExtractionConfig = ExtractionConfig()) -> dict:
    """Build an immutable versioned training snapshot from all datasets."""
    frames, lineage_all = {}, []
    for name in GOLD_DATASETS:
        df, lineage = extract_dataset(name, cfg)
        frames[name] = df
        lineage_all.append(lineage)
    snapshot_id = hashlib.sha256(
        "|".join(sorted(f"{l['dataset']}:{l['dataset_version']}" for l in lineage_all))
        .encode()).hexdigest()[:16]
    out = Path(cfg.snapshot_root) / snapshot_id
    out.mkdir(parents=True, exist_ok=True)
    for name, df in frames.items():
        df.to_parquet(out / f"{name}.parquet", index=False)
    manifest = {"snapshot_id": snapshot_id, "datasets": lineage_all}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[snapshot] {snapshot_id} -> {out}")
    for l in lineage_all:
        print(f"  {l['dataset']}: source={l['source']} rows={l['rows']} "
              f"version={l['dataset_version']}")
    return manifest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--min-rows", type=int, default=10_000)
    p.add_argument("--snapshot-root", default="data/snapshots")
    p.add_argument("--synthetic-seed", type=int, default=20240)
    args = p.parse_args()
    build_training_snapshot(ExtractionConfig(min_rows=args.min_rows,
                                             synthetic_seed=args.synthetic_seed,
                                             snapshot_root=args.snapshot_root))


if __name__ == "__main__":
    main()
