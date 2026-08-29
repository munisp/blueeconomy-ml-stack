"""Registry/run-log consistency tests (ML-1 part c).

Every models/<name>/<semver>/ directory must be backed by a real training
run in results/runs.jsonl with a matching metrics hash, and stage pointers
must reference existing versions.
"""

import json
from pathlib import Path

from pipelines.registry_check import check_registry

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_repo_registry_is_consistent():
    violations = check_registry(REPO_ROOT / "models",
                                REPO_ROOT / "results" / "runs.jsonl")
    assert violations == [], "\n".join(violations)


def _seed_registry(tmp_path, auroc=0.9, run_auroc=None, with_run=True):
    models_root = tmp_path / "models"
    vdir = models_root / "demo-model" / "0.1.0"
    vdir.mkdir(parents=True)
    (vdir / "metrics.json").write_text(json.dumps({
        "model": "demo-model", "version": "0.1.0",
        "test": {"auroc": auroc}}))
    runs = tmp_path / "runs.jsonl"
    entries = []
    if with_run:
        entries.append({
            "params": {"model": "demo-model", "version": "0.1.0"},
            "metrics": {"test_auroc": auroc if run_auroc is None else run_auroc}})
    runs.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    return models_root, runs


def test_version_dir_without_run_entry_flagged(tmp_path):
    models_root, runs = _seed_registry(tmp_path, with_run=False)
    violations = check_registry(models_root, runs)
    assert any("no training run" in v for v in violations)


def test_metrics_hash_mismatch_flagged(tmp_path):
    models_root, runs = _seed_registry(tmp_path, auroc=0.9, run_auroc=0.8)
    violations = check_registry(models_root, runs)
    assert any("metrics hash mismatch" in v for v in violations)


def test_dangling_production_pointer_flagged(tmp_path):
    models_root, runs = _seed_registry(tmp_path)
    (models_root / "demo-model" / "PRODUCTION").write_text("9.9.9\n")
    violations = check_registry(models_root, runs)
    assert any("pointer references missing version" in v for v in violations)


def test_consistent_fixture_passes(tmp_path):
    models_root, runs = _seed_registry(tmp_path)
    (models_root / "demo-model" / "PRODUCTION").write_text("0.1.0\n")
    assert check_registry(models_root, runs) == []
