"""Fail-closed inference contract tests.

These tests encode the platform doctrine: a missing/invalid model or a
malformed request must yield SCORING_UNAVAILABLE + rules_only, NEVER a
fabricated score. They use the committed 0.1.0 artifacts when present and
skip (not fake) the live-model assertions otherwise.
"""

from pathlib import Path

import pytest

from inference.scoring import STATUS_OK, STATUS_UNAVAILABLE, Scorer

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"


def test_missing_model_is_unavailable_not_fabricated(tmp_path):
    scorer = Scorer(tmp_path, "declaration-fraud", ["9.9.9"])
    r = scorer.score([0.0] * 11, entity_id="TIN-1")
    assert r.status == STATUS_UNAVAILABLE
    assert r.score is None
    assert r.mode == "rules_only"


def test_garbage_model_file_is_unavailable(tmp_path):
    d = tmp_path / "declaration-fraud" / "0.0.0"
    d.mkdir(parents=True)
    (d / "model.onnx").write_bytes(b"not an onnx file")
    scorer = Scorer(tmp_path, "declaration-fraud", ["0.0.0"])
    r = scorer.score([0.0] * 11, entity_id="TIN-1")
    assert r.status == STATUS_UNAVAILABLE and r.score is None


def test_unknown_version_routes_fail_closed(tmp_path):
    scorer = Scorer(tmp_path, "declaration-fraud", ["1.0.0", "2.0.0"])
    for i in range(20):
        r = scorer.score([0.0] * 11, entity_id=f"e-{i}")
        assert r.status == STATUS_UNAVAILABLE and r.score is None


@pytest.mark.skipif(not (MODELS / "declaration-fraud" / "0.1.0" / "model.onnx").is_file(),
                    reason="trained 0.1.0 artifacts not present")
def test_committed_model_scores_on_cpu():
    scorer = Scorer(MODELS, "declaration-fraud", ["0.1.0"])
    r = scorer.score([0.4, 10.0, 0.2, 8.0, 1, 0, 1, 1, 0, 2.0, 3.0], entity_id="TIN-9")
    assert r.status == STATUS_OK
    assert r.score is not None and 0.0 <= r.score <= 1.0
    assert r.latency_ms < 50.0  # CPU latency budget


@pytest.mark.skipif(not (MODELS / "declaration-fraud" / "0.1.0" / "model.onnx").is_file(),
                    reason="trained 0.1.0 artifacts not present")
def test_wrong_feature_count_fails_closed():
    scorer = Scorer(MODELS, "declaration-fraud", ["0.1.0"])
    r = scorer.score([0.0] * 5, entity_id="TIN-9")
    assert r.status == STATUS_UNAVAILABLE and r.score is None
