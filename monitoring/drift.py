"""Drift detection: feature drift + prediction drift reports.

Uses Evidently (Apache-2.0) when installed; degrades to a PSI (population
stability index) implementation so the scheduled job always produces an
honest drift artifact. Reports are written as versioned JSON artifacts — they
are scheduled-job outputs, never blocking gates.

Usage:
    python -m monitoring.drift --reference data/synthetic/declarations.parquet \
        --current data/synthetic/declarations_shifted.parquet \
        --out results/drift/declaration_drift.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from evidently import Report
    from evidently.presets import DataDriftPreset
    _HAS_EVIDENTLY = True
except ImportError:  # pragma: no cover
    _HAS_EVIDENTLY = False


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population stability index between two 1-D samples."""
    quantiles = np.linspace(0, 100, bins + 1)
    edges = np.unique(np.percentile(reference, quantiles))
    if len(edges) < 3:
        return 0.0
    r = np.clip(np.histogram(reference, edges)[0] / max(1, len(reference)), 1e-4, None)
    c = np.clip(np.histogram(current, edges)[0] / max(1, len(current)), 1e-4, None)
    return float(np.sum((r - c) * np.log(r / c)))


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
            and c not in {"is_fraud", "node_label"}]


def compute_drift(reference: pd.DataFrame, current: pd.DataFrame,
                  out_path: str | Path, prediction_col: str | None = None) -> dict:
    cols = [c for c in numeric_columns(reference) if c in current.columns]
    result = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "backend": "evidently" if _HAS_EVIDENTLY else "psi-fallback",
              "n_reference": len(reference), "n_current": len(current),
              "feature_drift": {}, "prediction_drift": None}

    if _HAS_EVIDENTLY:
        import re
        report = Report(metrics=[DataDriftPreset()])
        snap = report.run(reference_data=reference[cols],
                          current_data=current[cols])
        payload = json.loads(snap.json())
        pat = re.compile(r"ValueDrift\(column=(?P<col>[^,]+),.*threshold=(?P<thr>[0-9.]+)\)")
        for m in payload.get("metrics", []):
            match = pat.match(m.get("metric_name", ""))
            if not match:
                continue
            thr = float(match.group("thr"))
            score = float(m["value"])
            result["feature_drift"][match.group("col")] = {
                "drift_score": score, "threshold": thr,
                "drift_detected": bool(score >= thr),
            }
    else:
        for c in cols:
            score = psi(reference[c].to_numpy(dtype=float),
                        current[c].to_numpy(dtype=float))
            result["feature_drift"][c] = {
                "drift_score": score,
                "drift_detected": bool(score > 0.2),  # common PSI threshold
            }

    if prediction_col and prediction_col in reference and prediction_col in current:
        score = psi(reference[prediction_col].to_numpy(dtype=float),
                    current[prediction_col].to_numpy(dtype=float))
        result["prediction_drift"] = {"column": prediction_col,
                                      "drift_score": score,
                                      "drift_detected": bool(score > 0.2)}

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"[drift] backend={result['backend']} features={len(cols)} -> {out}")
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--reference", required=True)
    p.add_argument("--current", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--prediction-col", default=None)
    args = p.parse_args()
    compute_drift(pd.read_parquet(args.reference), pd.read_parquet(args.current),
                  args.out, prediction_col=args.prediction_col)


if __name__ == "__main__":
    main()
