"""Registry/run-log consistency check (CI-scriptable, fail-closed).

For every models/<name>/<semver>/ directory, require:
  1. a metrics.json next to the artifacts;
  2. a results/runs.jsonl training-run entry whose params.model /
     params.version match the metrics.json identity (no fabricated or
     run-less promotions — ML-1: 0.1.1 was "promoted" with no training run);
  3. a matching METRICS HASH: the sha256 of the canonical flattened test
     metrics in metrics.json must equal the hash of the corresponding
     test_*/baseline_test_* metrics recorded in the run entry, so registry
     metrics cannot drift from the logged run evidence;
  4. PRODUCTION / Staging pointer files must reference an existing version
     directory.

Exit code 0 = consistent, 1 = violations found (details on stdout).

Usage:
    python -m pipelines.registry_check [--models-root models] \
        [--runs results/runs.jsonl]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from pipelines.continuous_training import _SEMVER

POINTERS = ("PRODUCTION", "Staging")


def _flatten_recorded_metrics(metrics_json: dict) -> dict[str, float]:
    """Flatten a registry metrics.json into runs.jsonl-style keys.

    "test" section            -> test_<metric>
    "baseline_<algo>_test"    -> baseline_test_<metric>
    """
    flat: dict[str, float] = {}
    for section, values in metrics_json.items():
        if not isinstance(values, dict):
            continue
        if section == "test":
            prefix = "test_"
        elif section.startswith("baseline_") and section.endswith("_test"):
            prefix = "baseline_test_"
        else:
            continue
        for key, value in values.items():
            flat[prefix + key] = value
    return flat


def _canonical_hash(flat: dict) -> str:
    canon = json.dumps(flat, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()


def _run_metrics_hash(run: dict) -> str:
    flat = {k: v for k, v in run.get("metrics", {}).items()
            if k.startswith("test_") or k.startswith("baseline_test_")}
    return _canonical_hash(flat)


def _load_runs(runs_path: Path) -> list[dict]:
    runs = []
    with open(runs_path) as f:
        for line in f:
            line = line.strip()
            if line:
                runs.append(json.loads(line))
    return runs


def check_registry(models_root: Path, runs_path: Path) -> list[str]:
    """Return a list of violation strings (empty = consistent)."""
    violations: list[str] = []
    models_root = Path(models_root)
    runs_path = Path(runs_path)
    if not models_root.is_dir():
        return [f"models root missing: {models_root}"]
    if not runs_path.is_file():
        return [f"runs log missing: {runs_path}"]
    runs = _load_runs(runs_path)

    for model_dir in sorted(p for p in models_root.iterdir() if p.is_dir()):
        version_dirs = []
        for child in sorted(model_dir.iterdir()):
            if child.is_dir():
                if not _SEMVER.match(child.name):
                    violations.append(
                        f"{child}: directory is not a semver version")
                    continue
                version_dirs.append(child)
            elif child.name in POINTERS:
                target = child.read_text().strip()
                if not (model_dir / target).is_dir():
                    violations.append(
                        f"{child}: pointer references missing version "
                        f"'{target}'")
        for vdir in version_dirs:
            metrics_file = vdir / "metrics.json"
            if not metrics_file.is_file():
                violations.append(f"{vdir}: metrics.json missing")
                continue
            try:
                recorded = json.loads(metrics_file.read_text())
            except json.JSONDecodeError as exc:
                violations.append(f"{metrics_file}: unreadable JSON: {exc}")
                continue
            identity = (recorded.get("model"), recorded.get("version"))
            if identity[1] != vdir.name:
                violations.append(
                    f"{metrics_file}: version field '{identity[1]}' does not "
                    f"match directory '{vdir.name}'")
            matches = [r for r in runs
                       if r.get("params", {}).get("model") == identity[0]
                       and r.get("params", {}).get("version") == identity[1]]
            if not matches:
                violations.append(
                    f"{vdir}: no training run in {runs_path} for "
                    f"model='{identity[0]}' version='{identity[1]}' "
                    f"(registry artifacts must come from a logged run)")
                continue
            recorded_hash = _canonical_hash(
                _flatten_recorded_metrics(recorded))
            if not any(_run_metrics_hash(r) == recorded_hash for r in matches):
                violations.append(
                    f"{vdir}: metrics hash mismatch — registry metrics.json "
                    f"(sha256 {recorded_hash[:12]}…) does not match any "
                    f"logged run's metrics for {identity[0]} {identity[1]}")
    return violations


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--models-root", default="models")
    p.add_argument("--runs", default="results/runs.jsonl")
    args = p.parse_args()
    violations = check_registry(Path(args.models_root), Path(args.runs))
    if violations:
        print(f"[registry-check] FAIL — {len(violations)} violation(s):")
        for v in violations:
            print(f"  - {v}")
        return 1
    print("[registry-check] OK — every registered version is backed by a "
          "logged training run with a matching metrics hash; pointers valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
