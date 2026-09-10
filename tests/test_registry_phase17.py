"""Phase-17 registry checks: G6 graph-mule-gnn served, G7 port-congestion
fail-closed until trained from real observations, #10 vessel-anomaly
regression guard."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from inference.scoring import STATUS_OK, STATUS_UNAVAILABLE, Scorer
from inference.service import MODEL_REGISTRY

MODELS_ROOT = Path(__file__).resolve().parent.parent / "models"


def test_registry_entries():
    for key in ("declaration-fraud", "vessel-anomaly", "port-congestion"):
        assert key in MODEL_REGISTRY


def test_graph_mule_gnn_exclusion_is_honest():
    # G6: the GNN artifact bakes its frozen training context graph into the
    # ONNX graph; the per-entity Scorer contract cannot execute it, so it
    # must NOT be in the serving registry (documented exclusion, not a
    # silent SCORING_UNAVAILABLE masquerading as a servable model).
    assert "graph-mule-gnn" not in MODEL_REGISTRY
    assert (MODELS_ROOT / "graph-mule-gnn" / "0.1.0" / "model.onnx").is_file()
    scorer = Scorer(MODELS_ROOT, "graph-mule-gnn", ["0.1.0"])
    result = scorer.score([0.0] * 9, entity_id="company-1")
    assert result.status == STATUS_UNAVAILABLE  # single-row scoring cannot run


def test_vessel_anomaly_served():
    scorer = Scorer(MODELS_ROOT, "vessel-anomaly", ["0.1.0"])
    assert scorer._load("0.1.0") is not None
    n = scorer._load("0.1.0").n_features
    result = scorer.score([0.0] * n, entity_id="mmsi-205123000")
    assert result.status == STATUS_OK, result.detail


def test_port_congestion_fail_closed_until_trained():
    # No committed artifact -> honest SCORING_UNAVAILABLE, never a heuristic.
    scorer = Scorer(MODELS_ROOT, "port-congestion", ["0.1.0"])
    result = scorer.score([0.0] * 10, entity_id="port-KEMBA")
    assert result.status == STATUS_UNAVAILABLE
    assert result.score is None
    assert result.mode == "rules_only"


def _synthetic_observations(n_per_port: int = 40) -> pd.DataFrame:
    rows = []
    t0 = pd.Timestamp("2026-01-01T00:00:00Z")
    for port, base in (("KEMBA", 5), ("TZDAR", 8)):
        for i in range(n_per_port):
            rows.append({
                "port_code": port,
                "queue_length": float(base + int(3 * np.sin(i / 4))),
                "observed_at": t0 + pd.Timedelta(minutes=30 * i),
                "data_source": "REAL:port_queue_observations",
            })
    return pd.DataFrame(rows)


def test_congestion_supervised_build_is_honest():
    from training.congestion import FEATURES, build_supervised

    df = _synthetic_observations()
    x, y, port_index = build_supervised(df, horizon_minutes=60)
    assert x.shape[1] == len(FEATURES)
    assert len(x) > 0 and len(x) == len(y)
    # The last anchor of each port has no future observation inside the
    # horizon window and must be dropped, not imputed.
    assert len(x) < len(df)
    assert set(port_index) == {"KEMBA", "TZDAR"}
    # Targets are real observed queue lengths from the same port series.
    assert y.min() >= 0


def test_congestion_training_disabled_without_dsn(monkeypatch):
    from training import congestion

    monkeypatch.delenv("BEML_CONGESTION_PG_DSN", raising=False)

    class Args:
        seed = 42
        device = "cpu"
        out = "models/port-congestion"
        version = "0.0.0-test"
        horizon_minutes = 60
        min_samples = 500
        hidden = 32
        epochs = 1

    with pytest.raises(SystemExit, match="config-gated"):
        congestion.train(Args())
