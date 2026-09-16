"""Phase-18 offline RL tests — synthetic fixtures live HERE, never in the
training path.

Covered:
- LinUCB bandit convergence on toy logged data + OPE gate promotion
- OPE gate refusal when the candidate cannot beat the logged baseline
  (and nothing is exported)
- registry honesty: RL models registered but SCORING_UNAVAILABLE with no
  committed artifacts; shadow-mode contract (mode/policy_version/
  autonomous=False), never an autonomous claim
- ONNX round-trip of a promoted policy through the production Scorer
- CQL trainer smoke + InsufficientHistory fail-closed behavior
"""

import argparse
import base64
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from inference.scoring import STATUS_OK, STATUS_UNAVAILABLE, Scorer
from training.rl import data as rl_data
from training.rl import ope as rl_ope
from training.rl import train as rl_train
from training.rl.bandits import LinUCB
from training.rl.cql import greedy_policy_actions, train_cql

MODELS_ROOT = Path(__file__).resolve().parent.parent / "models"
RL_MODELS = ("berth-allocation", "queue-policy", "route-advice")


# ---------------------------------------------------------------- toy data

# Analytic optimum of the toy world (clearance-latency reward, the only
# outcome that honestly exists in the declaration schema):
#   latency_hours(GREEN)  = 4 + 0.10 * risk_score   (fast lane re-queued
#                                                    when risky)
#   latency_hours(YELLOW) = 7
#   latency_hours(RED)    = 12
# reward = -0.05 * latency, so GREEN is optimal iff 4 + 0.1*risk < 7.
OPTIMAL_RISK_THRESHOLD = 30.0  # risk_score scale is 0..100 (0012 CHECK)


def _toy_queue_frame(n: int = 300, seed: int = 0,
                     behavior: str = "random") -> pd.DataFrame:
    """Synthetic customs_declarations-shaped history (TESTS ONLY).

    Columns match port-interop db/migrations/0012_declarations.sql exactly
    (submitted_at/cleared_at/is_aeo/risk_score/invoice_amount_minor/
    risk_lane). Ground truth is linear-in-context by construction so
    LinUCB can represent it (see OPTIMAL_RISK_THRESHOLD above).
    """
    rng = np.random.default_rng(seed)
    risk = rng.uniform(0, 100, n)
    lanes = np.array(["GREEN", "YELLOW", "RED"])
    if behavior == "random":
        lane = lanes[rng.integers(0, 3, n)]
    else:  # optimal logging policy
        lane = np.where(risk < OPTIMAL_RISK_THRESHOLD, "GREEN", "YELLOW")
    latency_h = np.where(lane == "GREEN", 4.0 + 0.10 * risk,
                         np.where(lane == "YELLOW", 7.0, 12.0))
    t0 = pd.Timestamp("2026-01-01T00:00:00Z")
    submitted = [t0 + pd.Timedelta(hours=int(i)) for i in range(n)]
    cleared = [s + pd.Timedelta(hours=float(l))
               for s, l in zip(submitted, latency_h)]
    return pd.DataFrame({
        "declaration_id": [f"D-{i}" for i in range(n)],
        "submitted_at": submitted,
        "cleared_at": cleared,
        "is_aeo": rng.integers(0, 2, n).astype(bool),
        "risk_score": risk,
        # minor units (cents), as in the real schema
        "invoice_amount_minor": (rng.uniform(100, 10000, n) * 100).astype(np.int64),
        "risk_lane": lane,
    })


def _toy_berth_frame(n: int = 60, seed: int = 1) -> pd.DataFrame:
    """Synthetic recommendation_log (kind=berth_allocation) frame with
    realized outcomes, shaped like rl_data.BERTH_QUERY output."""
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2026-01-01T00:00:00Z")
    berth = np.where(rng.integers(0, 2, n) == 0, "B-1", "B-2")
    return pd.DataFrame({
        "recommendation_id": [f"R-{i}" for i in range(n)],
        "created_at": [t0 + pd.Timedelta(hours=i) for i in range(n)],
        "port_code": "NGAPP",
        "n_vessels": rng.integers(1, 6, n),
        "n_berths": 2,
        "berth_id": berth,
        "waiting_hours": np.where(berth == "B-1", 2.0, 6.0),
        "turnaround_hours": np.where(berth == "B-1", 20.0, 18.0),
    })


def _toy_route_frame(n: int = 60, seed: int = 2) -> pd.DataFrame:
    """Synthetic recommendation_log (kind=route_advice) frame with realized
    outcomes, shaped like rl_data.ROUTE_QUERY output."""
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2026-01-01T00:00:00Z")
    option = rng.integers(0, 2, n)
    return pd.DataFrame({
        "recommendation_id": [f"R-{i}" for i in range(n)],
        "created_at": [t0 + pd.Timedelta(hours=i) for i in range(n)],
        "origin": "NGAPP",
        "destination": "GHTEM",
        "route_option": option,
        "predicted_delay_min": rng.uniform(10, 120, n),
        "realized_delay_min": np.where(option == 0, 30.0, 90.0),
        "hour": rng.integers(0, 24, n).astype(float),
        "dow": rng.integers(0, 7, n).astype(float),
    })


def _args(task: str, out: Path, **over):
    defaults = dict(task=task, out=str(out), version="0.1.0", min_samples=20,
                    train_frac=0.8, min_delta=0.0, alpha_ucb=1.0,
                    alpha_cql=1.0, gamma=0.0, epochs=60, seed=42)
    defaults.update(over)
    return argparse.Namespace(**defaults)


# ------------------------------------------------------- bandit convergence

def test_linucb_converges_on_toy_data():
    replay = rl_data.build_queue_replay(
        _toy_queue_frame(), min_samples=20,
        outcome_source=rl_data.QUEUE_OUTCOME_CLEARANCE_LATENCY)
    bandit = LinUCB(replay.n_actions, replay.contexts.shape[1]).fit(
        replay.contexts, replay.actions, replay.rewards)
    acts = bandit.greedy_actions(replay.contexts)
    risk = replay.contexts[:, 0]
    # GREEN (0) is optimal below the threshold, YELLOW (1) above it.
    optimal = np.where(risk < OPTIMAL_RISK_THRESHOLD, 0, 1)
    assert (acts == optimal).mean() > 0.85


def test_ope_beats_bad_logging_policy():
    replay = rl_data.build_queue_replay(
        _toy_queue_frame(), min_samples=20,
        outcome_source=rl_data.QUEUE_OUTCOME_CLEARANCE_LATENCY)
    bandit = LinUCB(replay.n_actions, replay.contexts.shape[1]).fit(
        replay.contexts, replay.actions, replay.rewards)
    cand = bandit.greedy_actions(replay.contexts)
    report = rl_ope.evaluate(cand, replay.contexts, replay.actions,
                             replay.rewards, replay.n_actions)
    assert report.promoted, report.reason
    assert report.dr_candidate > report.baseline_mean_reward


# ---------------------------------------------------------- OPE gate refusal

def test_ope_gate_refuses_when_candidate_cannot_beat_baseline():
    # Logging policy is optimal; the candidate deliberately inverts it
    # (manual-review exactly where it is harmful). DR must estimate the
    # candidate far below the logged baseline and the gate must refuse.
    replay = rl_data.build_queue_replay(
        _toy_queue_frame(behavior="optimal"), min_samples=20,
        outcome_source=rl_data.QUEUE_OUTCOME_CLEARANCE_LATENCY)
    risk = replay.contexts[:, 0]
    cand = np.where(risk < OPTIMAL_RISK_THRESHOLD, 1, 0)  # exact inversion
    report = rl_ope.evaluate(cand, replay.contexts, replay.actions,
                             replay.rewards, replay.n_actions)
    assert not report.promoted
    assert "overlap failure" in report.reason


def test_train_refuses_export_when_ope_gate_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("BEML_RL_QUEUE_PG_DSN", "postgres://unused-in-test")
    monkeypatch.setenv("BEML_RL_QUEUE_OUTCOME_SOURCE", "clearance-latency")
    monkeypatch.setattr(rl_data, "load_frame",
                        lambda dsn, q: _toy_queue_frame(behavior="optimal"))
    out = tmp_path / "queue-policy"
    with pytest.raises(SystemExit, match="OPE_GATE_REFUSED"):
        rl_train.train(_args("queue-policy", out, min_delta=0.01))
    assert not out.exists()  # nothing exported on refusal


def test_train_honest_exit_without_dsn(monkeypatch, tmp_path):
    monkeypatch.delenv("BEML_RL_QUEUE_PG_DSN", raising=False)
    with pytest.raises(SystemExit, match="BEML_RL_QUEUE_PG_DSN"):
        rl_train.train(_args("queue-policy", tmp_path / "qp"))


def test_insufficient_history_is_fail_closed():
    with pytest.raises(rl_data.InsufficientHistory, match="INSUFFICIENT_HISTORY"):
        rl_data.build_queue_replay(_toy_queue_frame(n=10), min_samples=500,
                                   outcome_source="clearance-latency")


# -------------------------------------------- C1: schema + outcome honesty

def test_queue_query_targets_real_schema():
    """C1 regression: the replay query must reference the REAL
    customs_declarations columns (port-interop 0012), never the fabricated
    declaration_queue_history / hit / aeo_status / declared_value shape."""
    q = rl_data.QUEUE_QUERY.lower()
    assert "from customs_declarations" in q
    for col in ("submitted_at", "cleared_at", "is_aeo", "risk_score",
                "invoice_amount_minor", "risk_lane"):
        assert col in q
    for fabricated in ("declaration_queue_history", "hit", "aeo_status",
                       "declared_value", "priority_bucket", "processed_at"):
        assert fabricated not in q


def test_queue_replay_requires_configured_outcome_source():
    """No inspection-outcome column exists in any platform schema, so the
    reward builder must refuse to run without an explicit outcome source."""
    with pytest.raises(rl_data.InsufficientHistory, match="outcome source"):
        rl_data.build_queue_replay(_toy_queue_frame(), min_samples=20)
    with pytest.raises(rl_data.InsufficientHistory, match="outcome source"):
        rl_data.build_queue_replay(_toy_queue_frame(), min_samples=20,
                                   outcome_source="inspection-hits")


def test_queue_replay_maps_real_columns():
    replay = rl_data.build_queue_replay(
        _toy_queue_frame(), min_samples=20,
        outcome_source="clearance-latency")
    assert replay.data_source == "REAL:customs_declarations"
    assert replay.n_actions == 3
    assert set(np.unique(replay.actions)) <= {0, 1, 2}  # GREEN/YELLOW/RED
    # reward = -0.05 * latency_hours, always <= 0 (no fabricated hit term)
    assert (replay.rewards <= 0).all()
    # invoice minor units converted to major before log1p
    assert (replay.contexts[:, 2] >= 0).all()


def test_queue_replay_refuses_unmapped_lane():
    df = _toy_queue_frame(n=30)
    df.loc[0, "risk_lane"] = "BLUE"
    with pytest.raises(rl_data.InsufficientHistory, match="unmapped risk_lane"):
        rl_data.build_queue_replay(df, min_samples=20,
                                   outcome_source="clearance-latency")


def test_train_queue_honest_exit_without_outcome_source(monkeypatch, tmp_path):
    monkeypatch.setenv("BEML_RL_QUEUE_PG_DSN", "postgres://unused-in-test")
    monkeypatch.delenv("BEML_RL_QUEUE_OUTCOME_SOURCE", raising=False)
    monkeypatch.setattr(rl_data, "load_frame",
                        lambda dsn, q: _toy_queue_frame())
    with pytest.raises(rl_data.InsufficientHistory, match="outcome source"):
        rl_train.train(_args("queue-policy", tmp_path / "qp"))
    assert not (tmp_path / "qp").exists()


# -------------------------------------------- C2: berth replay vs real log

def test_berth_query_targets_real_schema():
    """C2 regression: no berths registry or port_calls lifecycle columns
    exist; the replay must read recommendation_log (geo 0018) JSONB."""
    q = rl_data.BERTH_QUERY.lower()
    assert "from recommendation_log" in q
    assert "kind = 'berth_allocation'" in q
    assert "outcome is not null" in q
    for fabricated in ("from port_calls", "join berths", "berthed_at",
                       "departed_at", "queue_len_at_arrival"):
        assert fabricated not in q


def test_berth_replay_builds_from_recommendation_log():
    replay = rl_data.build_berth_replay(_toy_berth_frame(), min_samples=20)
    assert replay.data_source == "REAL:recommendation_log"
    assert replay.n_actions == 2
    assert (replay.rewards <= 0).all()  # -(wait+turnaround)/24


def test_berth_replay_insufficient_without_outcomes():
    df = _toy_berth_frame(n=60)
    df["waiting_hours"] = None  # reward pipeline has not landed yet
    df["turnaround_hours"] = None
    with pytest.raises(rl_data.InsufficientHistory, match="INSUFFICIENT_HISTORY"):
        rl_data.build_berth_replay(df, min_samples=20)


# -------------------------------------------- C3: route replay vs real log

def test_route_query_targets_real_schema():
    """C3 regression: the replay must read recommendation_log (geo 0018)
    with action/reward extracted from suggestion/outcome JSONB."""
    q = rl_data.ROUTE_QUERY.lower()
    assert "from recommendation_log" in q
    assert "kind = 'route_advice'" in q
    assert "outcome is not null" in q
    assert "suggestion->>'routeoption'" in q
    assert "outcome->>'realizeddelaymin'" in q
    assert "route_advice_outcomes" not in q


def test_route_replay_builds_from_recommendation_log():
    replay = rl_data.build_route_replay(_toy_route_frame(), min_samples=20)
    assert replay.data_source == "REAL:recommendation_log"
    assert replay.n_actions == 2
    assert (replay.rewards <= 0).all()  # -realized_delay/60


# ------------------------------------------- promotion + ONNX round-trip

def test_promoted_policy_onnx_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("BEML_RL_QUEUE_PG_DSN", "postgres://unused-in-test")
    monkeypatch.setenv("BEML_RL_QUEUE_OUTCOME_SOURCE", "clearance-latency")
    monkeypatch.setattr(rl_data, "load_frame",
                        lambda dsn, q: _toy_queue_frame())
    out = tmp_path / "queue-policy"
    metrics = rl_train.train(_args("queue-policy", out))
    assert metrics["kind"] == "policy"
    assert metrics["ope"]["promoted"] is True
    assert metrics["data_source"] == "REAL:customs_declarations"
    assert (out / "0.1.0" / "model.onnx").is_file()
    assert (out / "0.1.0" / "metrics.json").is_file()

    scorer = Scorer(tmp_path, "queue-policy", ["0.1.0"])
    n_feat = scorer._load("0.1.0").n_features
    assert n_feat == len(metrics["features"])
    for risk in (0.9, 0.1):
        feats = [0.0] * n_feat
        feats[0] = risk
        r = scorer.score(feats, entity_id="decl-1")
        assert r.status == STATUS_OK, r.detail
        assert r.score in (0.0, 1.0, 2.0)  # action index, never a probability


def test_cql_trains_and_exports(tmp_path):
    rng = np.random.default_rng(1)
    n, d, k = 300, 5, 3
    x = rng.normal(0, 1, (n, d)).astype(np.float32)
    true_w = rng.normal(0, 1, (k, d)).astype(np.float32)
    a = rng.integers(0, k, n)
    r = (x * true_w[a]).sum(axis=1) + rng.normal(0, 0.01, n)
    qnet, meta = train_cql(x, a, r.astype(np.float32), k, epochs=300,
                           lr=1e-3, alpha=0.5)
    assert meta["gamma"] == 0.0
    acts = greedy_policy_actions(qnet, x)
    optimal = np.argmax(x @ true_w.T, axis=1)
    assert (acts == optimal).mean() > 0.85


# --------------------------------------------------------- registry honesty

def test_rl_models_registered_shadow():
    from inference.service import MODEL_REGISTRY
    for key in RL_MODELS:
        assert key in MODEL_REGISTRY
        assert MODEL_REGISTRY[key]["shadow"] is True


def test_rl_models_fail_closed_without_artifacts():
    for key in RL_MODELS:
        scorer = Scorer(MODELS_ROOT, key, ["0.1.0"])
        r = scorer.score([0.0] * 8, entity_id="e-1")
        assert r.status == STATUS_UNAVAILABLE
        assert r.score is None and r.mode == "rules_only"


# ------------------------------------------------ shadow-mode HTTP contract

def _b64u(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _service_client(tmp_path, monkeypatch, scorer=None):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from fastapi.testclient import TestClient
    from inference import service

    private = Ed25519PrivateKey.generate()
    public = private.public_key().public_bytes_raw()
    jwks = {"keys": [{"kty": "OKP", "crv": "Ed25519", "kid": "test-1",
                      "x": _b64u(public)}]}
    jwks_path = tmp_path / "jwks.json"
    jwks_path.write_text(json.dumps(jwks))
    monkeypatch.setenv("BEML_OIDC_JWKS_PATH", str(jwks_path))
    monkeypatch.setenv("BEML_OIDC_ISSUER",
                       "https://keycloak.example/realms/blueeconomy")
    if scorer is not None:
        monkeypatch.setitem(service.scorers, "queue-policy", scorer)
    client = TestClient(service.app)
    client.__enter__()  # run lifespan so the JWKS keyring loads

    def token():
        header = {"alg": "EdDSA", "kid": "test-1", "typ": "JWT"}
        payload = {"sub": "svc-tester",
                   "iss": "https://keycloak.example/realms/blueeconomy",
                   "exp": int(time.time()) + 300}
        si = f"{_b64u(json.dumps(header).encode())}.{_b64u(json.dumps(payload).encode())}"
        return f"{si}.{_b64u(private.sign(si.encode()))}"

    return client, token


def test_shadow_mode_unavailable_contract(tmp_path, monkeypatch):
    client, token = _service_client(tmp_path, monkeypatch)
    resp = client.post("/score/queue-policy",
                       json={"entity_id": "d-1", "features": [0.0] * 8},
                       headers={"Authorization": f"Bearer {token()}"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "SCORING_UNAVAILABLE"
    assert body["mode"] == "rules_only"
    assert body["autonomous"] is False
    assert "policy_version" in body


def test_shadow_mode_promoted_policy_contract(tmp_path, monkeypatch):
    monkeypatch.setenv("BEML_RL_QUEUE_PG_DSN", "postgres://unused-in-test")
    monkeypatch.setenv("BEML_RL_QUEUE_OUTCOME_SOURCE", "clearance-latency")
    monkeypatch.setattr(rl_data, "load_frame",
                        lambda dsn, q: _toy_queue_frame())
    models_root = tmp_path / "models"
    rl_train.train(_args("queue-policy", models_root / "queue-policy"))
    scorer = Scorer(models_root, "queue-policy", ["0.1.0"])
    client, token = _service_client(tmp_path, monkeypatch, scorer=scorer)
    resp = client.post("/score/queue-policy",
                       json={"entity_id": "d-1",
                             "features": [0.9, 0.0, 8.0, 0.0, 1.0, 0.0, 1.0, 1.0]},
                       headers={"Authorization": f"Bearer {token()}"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "OK"
    assert body["mode"] == "shadow"          # never "autonomous"/"ml"
    assert body["policy_version"] == "0.1.0"
    assert body["autonomous"] is False
    assert body["action_index"] in (0.0, 1.0, 2.0)
