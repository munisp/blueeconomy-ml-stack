"""/score authentication (Keycloak RS256) and production boot gates:
mandatory Keycloak coordinates, file-based MLFLOW_TRACKING_URI refusal."""

import base64
import importlib
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from inference.auth import (
    AuthError,
    BootError,
    KeycloakAuthenticator,
    KeycloakConfig,
    build_authenticator_from_env,
)
from training.common import BootError as TrackingBootError
from training.common import validate_tracking_uri

ISSUER = "https://keycloak.example/realms/blueeconomy"
AUDIENCE = "ml-scoring"


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _int_b64url(value: int) -> str:
    return _b64url(value.to_bytes((value.bit_length() + 7) // 8, "big"))


@pytest.fixture
def jwks_server():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    numbers = key.public_key().public_numbers()
    document = {
        "keys": [
            {
                "kid": "realm-key-1",
                "kty": "RSA",
                "alg": "RS256",
                "use": "sig",
                "n": _int_b64url(numbers.n),
                "e": _int_b64url(numbers.e),
            }
        ]
    }

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps(document).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield key, f"http://127.0.0.1:{server.server_port}/jwks"
    server.shutdown()
    thread.join()


def _sign(key, claims: dict, kid: str = "realm-key-1") -> str:
    header = _b64url(json.dumps({"alg": "RS256", "kid": kid, "typ": "JWT"}).encode())
    payload = _b64url(json.dumps(claims).encode())
    signature = key.sign(
        f"{header}.{payload}".encode(), padding.PKCS1v15(), hashes.SHA256()
    )
    return f"{header}.{payload}.{_b64url(signature)}"


def _claims(**overrides):
    claims = {
        "iss": ISSUER,
        "sub": "service-account-scoring-client",
        "aud": [AUDIENCE],
        "exp": int(time.time()) + 300,
        "iat": int(time.time()),
    }
    claims.update(overrides)
    return claims


@pytest.fixture
def authenticator(jwks_server):
    _, jwks_url = jwks_server
    return KeycloakAuthenticator(
        KeycloakConfig(jwks_url=jwks_url, issuer=ISSUER, audience=AUDIENCE)
    )


def test_valid_token_accepted(jwks_server, authenticator):
    key, _ = jwks_server
    authenticator.authenticate("Bearer " + _sign(key, _claims()))


@pytest.mark.parametrize(
    "claims",
    [
        _claims(exp=int(time.time()) - 60),  # expired
        {k: v for k, v in _claims().items() if k != "exp"},  # no expiry
        _claims(aud=["someone-else"]),  # wrong audience
        _claims(iss="https://evil.example/realms/blueeconomy"),  # wrong issuer
        _claims(sub=""),  # empty subject
        _claims(nbf=int(time.time()) + 600),  # not yet valid
    ],
)
def test_bad_claims_rejected(jwks_server, authenticator, claims):
    key, _ = jwks_server
    with pytest.raises(AuthError):
        authenticator.authenticate("Bearer " + _sign(key, claims))


def test_garbage_tokens_rejected(authenticator):
    for header in (None, "", "Basic abc", "Bearer", "Bearer not-a-jwt", "Bearer a.b.c"):
        with pytest.raises(AuthError):
            authenticator.authenticate(header)


def test_wrong_key_rejected(jwks_server, authenticator):
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    with pytest.raises(AuthError):
        authenticator.authenticate("Bearer " + _sign(other, _claims()))


def test_unreachable_jwks_refuses_startup():
    with pytest.raises(AuthError):
        KeycloakAuthenticator(
            KeycloakConfig(
                jwks_url="https://keys.invalid/jwks", issuer=ISSUER, audience=AUDIENCE
            )
        )


def test_production_requires_keycloak_coordinates():
    env = {"BEML_ENV": "production"}
    with pytest.raises(BootError, match="KEYCLOAK"):
        build_authenticator_from_env(env)
    # Unset BEML_ENV is production (fail-closed default).
    with pytest.raises(BootError, match="KEYCLOAK"):
        build_authenticator_from_env({})


def test_partial_keycloak_coordinates_refused():
    env = {
        "BEML_ENV": "development",
        "KEYCLOAK_JWKS_URL": "https://keys.example/jwks",
    }
    with pytest.raises(BootError, match="together"):
        build_authenticator_from_env(env)


def test_development_may_run_unauthenticated():
    assert build_authenticator_from_env({"BEML_ENV": "development"}) is None
    assert build_authenticator_from_env({"BEML_ENV": "test"}) is None


def test_development_with_coordinates_builds(jwks_server):
    _, jwks_url = jwks_server
    env = {
        "BEML_ENV": "development",
        "KEYCLOAK_JWKS_URL": jwks_url,
        "KEYCLOAK_ISSUER": ISSUER,
        "KEYCLOAK_EXPECTED_AUDIENCE": AUDIENCE,
    }
    assert build_authenticator_from_env(env) is not None


def test_production_refuses_file_based_tracking_uri():
    for uri in ("sqlite:///mlflow.db", "file:./mlruns", "mlflow.db", "./mlruns"):
        with pytest.raises(TrackingBootError, match="file-based"):
            validate_tracking_uri({"BEML_ENV": "production", "MLFLOW_TRACKING_URI": uri})
        with pytest.raises(TrackingBootError, match="file-based"):
            validate_tracking_uri({"MLFLOW_TRACKING_URI": uri})


def test_production_requires_tracking_uri_for_training():
    with pytest.raises(TrackingBootError, match="MLFLOW_TRACKING_URI"):
        validate_tracking_uri({"BEML_ENV": "production"})
    # The inference path records no runs and may leave it unset.
    validate_tracking_uri({"BEML_ENV": "production"}, required_in_production=False)


def test_production_accepts_pg_backed_tracking_uri():
    env = {"BEML_ENV": "production", "MLFLOW_TRACKING_URI": "http://mlflow:5000"}
    validate_tracking_uri(env)
    validate_tracking_uri(env, required_in_production=False)


def test_development_allows_file_tracking_uri():
    validate_tracking_uri({"BEML_ENV": "development", "MLFLOW_TRACKING_URI": "sqlite:///mlflow.db"})


def test_score_route_requires_token(jwks_server, monkeypatch):
    key, jwks_url = jwks_server
    monkeypatch.setenv("BEML_ENV", "development")
    monkeypatch.setenv("KEYCLOAK_JWKS_URL", jwks_url)
    monkeypatch.setenv("KEYCLOAK_ISSUER", ISSUER)
    monkeypatch.setenv("KEYCLOAK_EXPECTED_AUDIENCE", AUDIENCE)
    # Import the app only after the environment is wired; the service builds
    # its authenticator at import (boot) time.
    sys.modules.pop("inference.service", None)
    service = importlib.import_module("inference.service")
    from fastapi.testclient import TestClient

    client = TestClient(service.app)
    payload = {"entity_id": "TIN-1", "features": [0.0] * 11}
    assert client.post("/score/declaration-fraud", json=payload).status_code == 401
    assert (
        client.post(
            "/score/declaration-fraud",
            json=payload,
            headers={"Authorization": "Bearer garbage"},
        ).status_code
        == 401
    )
    token = _sign(key, _claims())
    response = client.post(
        "/score/declaration-fraud",
        json=payload,
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    # Health stays public.
    assert client.get("/health").status_code == 200
