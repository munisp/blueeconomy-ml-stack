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

# Analytic optimum of the toy world: E[reward | bucket=2] = 0.9*risk - 0.5
# (10h latency penalty), buckets 0/1 yield -0.25 (5h penalty, no hit), so
# manual-review is optimal iff risk > (0.5 - 0.25) / 0.9.
OPTIMAL_RISK_THRESHOLD = (0.5 - 0.25) / 0.9


def _toy_queue_frame(n: int = 300, seed: int = 0,
                     behavior: str = "random") -> pd.DataFrame:
    """Synthetic declaration queue history (TESTS ONLY).

    Ground truth is linear-in-context by construction so LinUCB can
    represent it: hit probability under manual-review is 0.9*risk_score.
    """
    rng = np.random.default_rng(seed)
    risk = rng.uniform(0, 1, n)
    if behavior == "random":
        bucket = rng.integers(0, 3, n)
    else:  # optimal logging policy
        bucket = np.where(risk > OPTIMAL_RISK_THRESHOLD, 2, 0)
    hit = ((bucket == 2) &
           (rng.uniform(0, 1, n) < 0.9 * risk)).astype(float)
    latency_h = np.where(bucket == 2, 10.0, 5.0)
    t0 = pd.Timestamp("2026-01-01T00:00:00Z")
    submitted = [t0 + pd.Timedelta(hours=int(i)) for i in range(n)]
    processed = [s + pd.Timedelta(hours=float(l))
                 for s, l in zip(submitted, latency_h)]
    return pd.DataFrame({
        "declaration_id": [f"D-{i}" for i in range(n)],
        "submitted_at": submitted,
        "processed_at": processed,
        "aeo_status": rng.integers(0, 2, n),
        "risk_score": risk,
        "declared_value": rng.uniform(100, 10000, n),
        "priority_bucket": bucket,
        "hit": hit,
    })


def _args(task: str, out: Path, **over):
    defaults = dict(task=task, out=str(out), version="0.1.0", min_samples=20,
                    train_frac=0.8, min_delta=0.0, alpha_ucb=1.0,
                    alpha_cql=1.0, gamma=0.0, epochs=60, seed=42)
    defaults.update(over)
    return argparse.Namespace(**defaults)


# ------------------------------------------------------- bandit convergence

def test_linucb_converges_on_toy_data():
    replay = rl_data.build_queue_replay(_toy_queue_frame(), min_samples=20)
    bandit = LinUCB(replay.n_actions, replay.contexts.shape[1]).fit(
        replay.contexts, replay.actions, replay.rewards)
    acts = bandit.greedy_actions(replay.contexts)
    risk = replay.contexts[:, 0]
    # low-risk rows may pick 0 or 1 (equal true reward); what matters is
    # that high-risk rows go to manual-review and low-risk rows do not
    agree = ((acts == 2) == (risk > OPTIMAL_RISK_THRESHOLD)).mean()
    assert agree > 0.85


def test_ope_beats_bad_logging_policy():
    replay = rl_data.build_queue_replay(_toy_queue_frame(), min_samples=20)
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
    replay = rl_data.build_queue_replay(_toy_queue_frame(behavior="optimal"),
                                        min_samples=20)
    risk = replay.contexts[:, 0]
    cand = np.where(risk > OPTIMAL_RISK_THRESHOLD, 0, 2)
    report = rl_ope.evaluate(cand, replay.contexts, replay.actions,
                             replay.rewards, replay.n_actions)
    assert not report.promoted
    assert "overlap failure" in report.reason


def test_train_refuses_export_when_ope_gate_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("BEML_RL_QUEUE_PG_DSN", "postgres://unused-in-test")
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
        rl_data.build_queue_replay(_toy_queue_frame(n=10), min_samples=500)


# ------------------------------------------- promotion + ONNX round-trip

def test_promoted_policy_onnx_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("BEML_RL_QUEUE_PG_DSN", "postgres://unused-in-test")
    monkeypatch.setattr(rl_data, "load_frame",
                        lambda dsn, q: _toy_queue_frame())
    out = tmp_path / "queue-policy"
    metrics = rl_train.train(_args("queue-policy", out))
    assert metrics["kind"] == "policy"
    assert metrics["ope"]["promoted"] is True
    assert metrics["data_source"] == "REAL:declaration_queue_history"
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
