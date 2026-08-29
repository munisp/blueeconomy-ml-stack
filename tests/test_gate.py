"""Evaluation-gate regression tests (ML-1).

ML-1: 0.1.1 was "promoted" through the first-model shortcut because no
PRODUCTION pointer existed for 0.1.0, so the improvement comparison was
skipped. The artifacts were a same-seed duplicate of 0.1.0 and a real
comparison would have rejected. These tests pin the hardened behaviour:

  (i)   re-running the same seed/weights as a new candidate -> REJECT
  (ii)  versions exist but no PRODUCTION pointer -> compare against the
        LATEST existing version; the comparison is never skipped
  (iii) a genuine first-ever model -> promote_first_model with a full record
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from pipelines.continuous_training import (evaluate_gate, notify, run_cycle,
                                           transition_stage)

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_010_METRICS = (REPO_ROOT / "models" / "declaration-fraud" / "0.1.0"
                    / "metrics.json")


def _write_metrics(model_dir: Path, version: str, auroc: float,
                   source_metrics: Path | None = None) -> Path:
    vdir = model_dir / version
    vdir.mkdir(parents=True, exist_ok=True)
    if source_metrics is not None:
        shutil.copy(source_metrics, vdir / "metrics.json")
    else:
        (vdir / "metrics.json").write_text(
            json.dumps({"test": {"auroc": auroc}}))
    return vdir


# (i) same-seed duplicate candidate -> gate REJECTS -------------------------

def test_same_seed_duplicate_candidate_rejected(tmp_path):
    """Replaying the ML-1 scenario: candidate metric == incumbent metric
    (same seed/weights, zero improvement) must be rejected."""
    model_dir = tmp_path / "declaration-fraud"
    _write_metrics(model_dir, "0.1.0", 0.9708187549580394)
    transition_stage(tmp_path, "declaration-fraud", "0.1.0", "PRODUCTION")
    # duplicate candidate: byte-identical metrics, as 0.1.1 was
    _write_metrics(model_dir, "0.1.1", 0.9708187549580394)

    promoted, gate = evaluate_gate(tmp_path, "declaration-fraud", "0.1.1")

    assert not promoted
    assert gate["decision"] == "reject"
    assert gate["incumbent"] == "0.1.0"
    assert gate["incumbent_metric"] == pytest.approx(0.9708187549580394)
    assert gate["candidate_metric"] == gate["incumbent_metric"]
    assert gate["candidate_metric"] < gate["incumbent_metric"] + gate["min_delta"]


def test_real_010_metrics_replayed_as_new_candidate_rejected(tmp_path):
    """Same as above but using the actual 0.1.0 metrics.json from the repo —
    this is exactly the evidence that the 0.1.1 'promotion' was fabricated."""
    model_dir = tmp_path / "declaration-fraud"
    _write_metrics(model_dir, "0.1.0", 0, source_metrics=REAL_010_METRICS)
    transition_stage(tmp_path, "declaration-fraud", "0.1.0", "PRODUCTION")
    _write_metrics(model_dir, "0.1.1", 0, source_metrics=REAL_010_METRICS)

    promoted, gate = evaluate_gate(tmp_path, "declaration-fraud", "0.1.1")

    assert not promoted
    assert gate["decision"] == "reject"


# (ii) no-pointer-but-versions-exist -> compare against latest ---------------

def test_no_pointer_compares_against_latest_version(tmp_path):
    """Versions exist but no PRODUCTION pointer: the gate must NOT take the
    first-model shortcut; it compares against the latest existing version."""
    model_dir = tmp_path / "declaration-fraud"
    _write_metrics(model_dir, "0.1.0", 0.90)
    _write_metrics(model_dir, "0.2.0", 0.95)  # latest, never deployed
    _write_metrics(model_dir, "0.3.0", 0.9505)  # beats 0.1.0 but not 0.2.0+delta

    promoted, gate = evaluate_gate(tmp_path, "declaration-fraud", "0.3.0")

    assert not promoted
    assert gate["decision"] == "reject"
    assert gate["incumbent"] == "0.2.0"
    assert gate["incumbent_source"] == "latest_version_no_pointer"
    assert gate["incumbent_metric"] == pytest.approx(0.95)


def test_no_pointer_candidate_beating_latest_promotes(tmp_path):
    model_dir = tmp_path / "declaration-fraud"
    _write_metrics(model_dir, "0.1.0", 0.90)
    _write_metrics(model_dir, "0.2.0", 0.95)
    _write_metrics(model_dir, "0.3.0", 0.952)  # >= 0.95 + 0.001

    promoted, gate = evaluate_gate(tmp_path, "declaration-fraud", "0.3.0")

    assert promoted
    assert gate["decision"] == "promote"
    assert gate["incumbent"] == "0.2.0"
    assert gate["incumbent_metric"] == pytest.approx(0.95)
    assert gate["candidate_metric"] == pytest.approx(0.952)


def test_dangling_pointer_falls_back_to_latest_and_records_it(tmp_path):
    model_dir = tmp_path / "declaration-fraud"
    _write_metrics(model_dir, "0.1.0", 0.90)
    transition_stage(tmp_path, "declaration-fraud", "9.9.9", "PRODUCTION")
    _write_metrics(model_dir, "0.2.0", 0.899)

    promoted, gate = evaluate_gate(tmp_path, "declaration-fraud", "0.2.0")

    assert not promoted
    assert gate["incumbent"] == "0.1.0"
    assert gate["incumbent_source"] == "latest_version_pointer_invalid"


def test_unreadable_incumbent_metrics_rejects_fail_closed(tmp_path):
    model_dir = tmp_path / "declaration-fraud"
    (model_dir / "0.1.0").mkdir(parents=True)
    (model_dir / "0.1.0" / "metrics.json").write_text("{not json")
    transition_stage(tmp_path, "declaration-fraud", "0.1.0", "PRODUCTION")
    _write_metrics(model_dir, "0.2.0", 0.99)

    promoted, gate = evaluate_gate(tmp_path, "declaration-fraud", "0.2.0")

    assert not promoted
    assert gate["decision"] == "reject"
    assert gate["reason"].startswith("incumbent_metrics_unreadable")


# (iii) genuine first-ever model ---------------------------------------------

def test_genuine_first_model_promoted_with_full_record(tmp_path):
    model_dir = tmp_path / "declaration-fraud"
    model_dir.mkdir(parents=True)
    _write_metrics(model_dir, "0.1.0", 0.91)

    promoted, gate = evaluate_gate(tmp_path, "declaration-fraud", "0.1.0")

    assert promoted
    assert gate["decision"] == "promote_first_model"
    assert gate["incumbent"] is None
    assert gate["incumbent_metric"] is None
    assert gate["candidate_metric"] == pytest.approx(0.91)
    assert gate["min_delta"] == pytest.approx(0.001)


# end-to-end through run_cycle (training/export subprocesses stubbed) --------

def _stub_subprocess(monkeypatch, candidate_metrics: dict):
    """Fake `python -m training.tabular ...` (writes candidate metrics) and
    `python -m inference.export_onnx ...` (no-op)."""
    def fake_run(cmd, check=False, **kwargs):
        if "training.tabular" in cmd:
            out = Path(cmd[cmd.index("--out") + 1])
            version = cmd[cmd.index("--version") + 1]
            vdir = out / version
            vdir.mkdir(parents=True, exist_ok=True)
            (vdir / "metrics.json").write_text(json.dumps(candidate_metrics))
        return subprocess.CompletedProcess(cmd, 0)
    monkeypatch.setattr(
        "pipelines.continuous_training.subprocess.run", fake_run)


def test_run_cycle_rejects_duplicate_candidate(tmp_path, monkeypatch):
    """Full cycle: a retrained same-metric candidate is rejected, the
    PRODUCTION pointer is left untouched, and the notification records the
    comparison (incumbent + both metrics + decision)."""
    model_dir = tmp_path / "models" / "declaration-fraud"
    _write_metrics(model_dir, "0.1.0", 0.97)
    transition_stage(tmp_path / "models", "declaration-fraud", "0.1.0",
                     "PRODUCTION")
    _stub_subprocess(monkeypatch, {"test": {"auroc": 0.97}})  # same seed
    outbox = tmp_path / "notifications.jsonl"
    monkeypatch.setattr("pipelines.continuous_training.notify",
                        lambda event, outbox=outbox: notify(event, outbox))

    result = run_cycle("declaration-fraud", data_dir=str(tmp_path),
                       candidate_version="0.1.1",
                       models_root=str(tmp_path / "models"))

    assert result["promoted"] is False
    assert result["gate"]["decision"] == "reject"
    assert result["gate"]["incumbent"] == "0.1.0"
    assert result["gate"]["incumbent_metric"] == pytest.approx(0.97)
    assert (model_dir / "PRODUCTION").read_text().strip() == "0.1.0"
    record = json.loads(outbox.read_text().strip().splitlines()[-1])
    assert record["gate"]["decision"] == "reject"
    assert record["gate"]["incumbent_metric"] == pytest.approx(0.97)
    assert record["gate"]["candidate_metric"] == pytest.approx(0.97)


def test_run_cycle_first_model_promotes_with_full_record(tmp_path, monkeypatch):
    (tmp_path / "models" / "declaration-fraud").mkdir(parents=True)
    _stub_subprocess(monkeypatch, {"test": {"auroc": 0.91}})
    outbox = tmp_path / "notifications.jsonl"
    monkeypatch.setattr("pipelines.continuous_training.notify",
                        lambda event, outbox=outbox: notify(event, outbox))

    result = run_cycle("declaration-fraud", data_dir=str(tmp_path),
                       candidate_version="0.1.0",
                       models_root=str(tmp_path / "models"))

    assert result["promoted"] is True
    assert result["gate"]["decision"] == "promote_first_model"
    assert result["gate"]["candidate_metric"] == pytest.approx(0.91)
    assert (tmp_path / "models" / "declaration-fraud" / "PRODUCTION"
            ).read_text().strip() == "0.1.0"
    record = json.loads(outbox.read_text().strip().splitlines()[-1])
    assert record["gate"]["decision"] == "promote_first_model"
    assert record["gate"]["candidate_metric"] == pytest.approx(0.91)
    assert record["gate"]["incumbent"] is None
