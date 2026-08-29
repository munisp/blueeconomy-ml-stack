"""Pipeline fail-closed + evaluation-gate tests."""

import json
import os
from pathlib import Path

import pytest

from pipelines.extract import (ExtractionConfig, LakehouseConfigurationError,
                               build_training_snapshot, extract_dataset)


def test_extract_fails_closed_without_fallback(monkeypatch, tmp_path):
    monkeypatch.delenv("BEML_LAKEHOUSE_ROOT", raising=False)
    monkeypatch.delenv("BEML_ALLOW_SYNTHETIC_FALLBACK", raising=False)
    with pytest.raises(LakehouseConfigurationError):
        extract_dataset("declarations", ExtractionConfig())


def test_synthetic_fallback_is_labelled(monkeypatch, tmp_path):
    monkeypatch.delenv("BEML_LAKEHOUSE_ROOT", raising=False)
    monkeypatch.setenv("BEML_ALLOW_SYNTHETIC_FALLBACK", "1")
    df, lineage = extract_dataset(
        "declarations", ExtractionConfig(min_rows=10, synthetic_seed=99))
    assert lineage["source"] == "SYNTHETIC_FALLBACK"
    assert (df["data_source"] == "SYNTHETIC").all()


def test_gate_rejects_non_improving_candidate(tmp_path):
    from pipelines.continuous_training import transition_stage, _get_metric
    model_dir = tmp_path / "declaration-fraud"
    (model_dir / "0.1.0").mkdir(parents=True)
    (model_dir / "0.2.0").mkdir(parents=True)
    (model_dir / "0.1.0" / "metrics.json").write_text(
        json.dumps({"test": {"auroc": 0.90}}))
    (model_dir / "0.2.0" / "metrics.json").write_text(
        json.dumps({"test": {"auroc": 0.89}}))
    transition_stage(tmp_path, "declaration-fraud", "0.1.0", "PRODUCTION")
    cand = _get_metric(json.loads((model_dir / "0.2.0" / "metrics.json").read_text()),
                       "test.auroc")
    inc = _get_metric(json.loads((model_dir / "0.1.0" / "metrics.json").read_text()),
                      "test.auroc")
    assert cand < inc + 0.001  # gate must reject: candidate does not beat incumbent
