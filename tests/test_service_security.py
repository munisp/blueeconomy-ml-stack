"""Security-hardening regression tests for the scoring HTTP surface (S1, S5):

- /score/*, /docs and /openapi.json require a verified Keycloak bearer token;
  unauthenticated calls get 401 and, when OIDC is not configured at all,
  503 (fail-closed — no anonymous fallback).
- Request bodies above BEML_MAX_BODY_BYTES get a clean 413.
- Every response carries HSTS / nosniff / DENY / no-referrer headers.
- /health stays public (probe convention).
"""

import base64
import json
import time

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi.testclient import TestClient

from inference import service

ISSUER = "https://keycloak.example/realms/blueeconomy"


def _b64u(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _make_jwks(tmp_path):
    private = Ed25519PrivateKey.generate()
    public = private.public_key().public_bytes_raw()
    jwks = {"keys": [{"kty": "OKP", "crv": "Ed25519", "kid": "test-1", "x": _b64u(public)}]}
    path = tmp_path / "jwks.json"
    path.write_text(json.dumps(jwks))
    return private, path


def _token(private, **overrides):
    header = {"alg": "EdDSA", "kid": "test-1", "typ": "JWT"}
    payload = {
        "sub": "svc-tester",
        "iss": ISSUER,
        "exp": int(time.time()) + 300,
        "realm_access": {"roles": ["ml-scorer"]},
    }
    payload.update(overrides)
    signing_input = f"{_b64u(json.dumps(header).encode())}.{_b64u(json.dumps(payload).encode())}"
    signature = private.sign(signing_input.encode("ascii"))
    return f"{signing_input}.{_b64u(signature)}"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    private, jwks_path = _make_jwks(tmp_path)
    monkeypatch.setenv("BEML_OIDC_JWKS_PATH", str(jwks_path))
    monkeypatch.setenv("BEML_OIDC_ISSUER", ISSUER)
    with TestClient(service.app) as c:
        c._test_private_key = private
        yield c


@pytest.fixture()
def client_no_oidc(monkeypatch):
    for var in ("BEML_OIDC_JWKS_PATH", "BEML_OIDC_JWKS_URL", "BEML_OIDC_ISSUER"):
        monkeypatch.delenv(var, raising=False)
    with TestClient(service.app) as c:
        yield c


def test_score_requires_bearer(client):
    resp = client.post("/score/vessel-anomaly", json={"entity_id": "e1", "features": [0.0]})
    assert resp.status_code == 401
    assert resp.json()["detail"]["reason"] == "missing-bearer"


def test_score_rejects_garbage_token(client):
    resp = client.post(
        "/score/vessel-anomaly",
        json={"entity_id": "e1", "features": [0.0]},
        headers={"Authorization": "Bearer not-a-jwt"},
    )
    assert resp.status_code == 401


def test_score_rejects_wrong_issuer(client):
    token = _token(client._test_private_key, iss="https://evil.example/realms/x")
    resp = client.post(
        "/score/vessel-anomaly",
        json={"entity_id": "e1", "features": [0.0]},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 401
    assert resp.json()["detail"]["reason"] == "issuer-mismatch"


def test_score_rejects_expired_token(client):
    token = _token(client._test_private_key, exp=int(time.time()) - 10)
    resp = client.post(
        "/score/vessel-anomaly",
        json={"entity_id": "e1", "features": [0.0]},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 401
    assert resp.json()["detail"]["reason"] == "token-expired"


def test_score_accepts_valid_token_fail_closed_scoring(client):
    token = _token(client._test_private_key)
    resp = client.post(
        "/score/nonexistent-model",
        json={"entity_id": "e1", "features": [0.0]},
        headers={"Authorization": f"Bearer {token}"},
    )
    # Authenticated: the fail-closed scoring contract (not an auth error).
    assert resp.status_code == 200
    assert resp.json()["status"] == "SCORING_UNAVAILABLE"


def test_score_fails_closed_503_when_oidc_unconfigured(client_no_oidc):
    resp = client_no_oidc.post("/score/vessel-anomaly",
                               json={"entity_id": "e1", "features": [0.0]})
    assert resp.status_code == 503
    assert resp.json()["detail"]["reason"] == "auth-oidc-unavailable"


def test_docs_and_openapi_require_auth(client):
    assert client.get("/docs").status_code == 401
    assert client.get("/openapi.json").status_code == 401
    token = _token(client._test_private_key)
    headers = {"Authorization": f"Bearer {token}"}
    assert client.get("/openapi.json", headers=headers).status_code == 200
    assert client.get("/docs", headers=headers).status_code == 200


def test_docs_and_openapi_fail_closed_without_oidc(client_no_oidc):
    assert client_no_oidc.get("/openapi.json").status_code == 503
    assert client_no_oidc.get("/docs").status_code == 503


def test_oversized_body_gets_clean_413(client):
    token = _token(client._test_private_key)
    big = {"entity_id": "e1", "features": [0.0], "pad": "x" * (service.MAX_BODY_BYTES + 1)}
    resp = client.post(
        "/score/vessel-anomaly", json=big, headers={"Authorization": f"Bearer {token}"}
    )
    assert resp.status_code == 413
    assert resp.json()["limit_bytes"] == service.MAX_BODY_BYTES


def test_security_headers_on_all_responses(client):
    token = _token(client._test_private_key)
    for resp in (
        client.get("/health"),
        client.post("/score/vessel-anomaly", json={"entity_id": "e1", "features": [0.0]}),
        client.get("/openapi.json", headers={"Authorization": f"Bearer {token}"}),
    ):
        assert resp.headers["Strict-Transport-Security"].startswith("max-age=")
        assert resp.headers["X-Content-Type-Options"] == "nosniff"
        assert resp.headers["X-Frame-Options"] == "DENY"
        assert resp.headers["Referrer-Policy"] == "no-referrer"


def test_health_stays_public(client_no_oidc):
    resp = client_no_oidc.get("/health")
    assert resp.status_code == 200
    assert "status" in resp.json()


def test_score_role_gate_config_gated(client, monkeypatch):
    """M5: when BEML_SCORE_REQUIRED_ROLE is set, tokens lacking the role get
    403; when unset any verified token passes (contract defines no role)."""
    body = {"entity_id": "e-1", "features": [0.0] * 11}
    # Unset: any verified identity is admitted (scoring itself fails closed).
    monkeypatch.delenv("BEML_SCORE_REQUIRED_ROLE", raising=False)
    r = client.post("/score/declaration-fraud", json=body,
                    headers={"Authorization": f"Bearer {_token(client._test_private_key)}"})
    assert r.status_code == 200
    # Set: a token without the role is refused before scoring.
    monkeypatch.setenv("BEML_SCORE_REQUIRED_ROLE", "ml-scorer")
    no_role = _token(client._test_private_key, realm_access={"roles": ["other"]})
    r = client.post("/score/declaration-fraud", json=body,
                    headers={"Authorization": f"Bearer {no_role}"})
    assert r.status_code == 403
    assert r.json()["detail"]["reason"] == "missing-role"
    # With the role: admitted.
    r = client.post("/score/declaration-fraud", json=body,
                    headers={"Authorization": f"Bearer {_token(client._test_private_key)}"})
    assert r.status_code == 200


def test_score_rejects_non_finite_features(client):
    """M4: NaN/Inf features are refused at the API boundary (422), never
    propagated into ONNX or the signed inference event."""
    # NaN is not strict JSON, so send a raw body (the permissive server-side
    # parser accepts it and the request validator must refuse it).
    r = client.post("/score/declaration-fraud",
                    content='{"entity_id": "e-1", "features": [0.0, NaN, 0.0]}',
                    headers={"Authorization": f"Bearer {_token(client._test_private_key)}",
                             "Content-Type": "application/json"})
    assert r.status_code == 422
