"""Keycloak RS256 JWT authentication for the scoring service.

Fail-closed contract (same as the rest of the estate):

- ``KEYCLOAK_JWKS_URL``, ``KEYCLOAK_ISSUER`` and
  ``KEYCLOAK_EXPECTED_AUDIENCE`` configure RS256 access-token verification
  against the realm JWKS (exact issuer, required expiry, aud/azp audience
  match, RSA keys >= 2048 bits, no JWKS redirects).
- In the production profile (``BEML_ENV=production`` or unset) all three are
  mandatory and the JWKS endpoint must be reachable at boot — the service
  refuses to start otherwise.
- A partially configured Keycloak coordinate set is a boot error in every
  profile.
- Only an explicitly non-production profile (``BEML_ENV=development`` /
  ``test``) may serve ``/score`` unauthenticated, with a loud warning.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from urllib.parse import urlparse

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric.rsa import (
    RSAPublicKey,
    RSAPublicNumbers,
)
from fastapi import HTTPException, Request

log = logging.getLogger(__name__)

JWKS_MIN_REFRESH_INTERVAL = 30.0  # seconds; bounds forged-kid refresh storms
JWKS_FETCH_TIMEOUT = 10.0  # seconds
MAX_JWKS_BYTES = 1 << 20
MAX_TOKEN_BYTES = 16 << 10


class AuthError(RuntimeError):
    """The bearer token is absent, malformed or unverifiable (HTTP 401)."""


class BootError(RuntimeError):
    """Fatal configuration error. The process must refuse to start."""


def production_profile(env: dict | None = None) -> bool:
    """BEML_ENV unset means production (fail-closed default)."""
    env = os.environ if env is None else env
    return (env.get("BEML_ENV") or "").strip().lower() in ("", "production")


def _b64url_decode(segment: str) -> bytes:
    padding = "=" * (-len(segment) % 4)
    try:
        return base64.urlsafe_b64decode(segment + padding)
    except (ValueError, base64.binascii.Error) as exc:
        raise AuthError("token segment is not valid base64url") from exc


@dataclass(frozen=True)
class KeycloakConfig:
    jwks_url: str
    issuer: str
    audience: str

    def validate(self) -> None:
        for name, value in (
            ("jwks_url", self.jwks_url),
            ("issuer", self.issuer),
            ("audience", self.audience),
        ):
            if not value or value != value.strip():
                raise ValueError(f"keycloak {name} is required and must be canonical")
        for name, value in (("jwks_url", self.jwks_url), ("issuer", self.issuer)):
            if value.startswith("https://"):
                continue
            # Plaintext HTTP is acceptable only towards a loopback peer
            # (in-pod test fixture or sidecar); anything remote must be TLS.
            if value.startswith("http://") and (urlparse(value).hostname or "") in (
                "127.0.0.1",
                "::1",
                "localhost",
            ):
                continue
            raise ValueError(f"keycloak {name} must use https")


class KeycloakAuthenticator:
    """Verifies Keycloak RS256 access tokens against the realm JWKS.

    Keys are cached and refreshed on unknown key IDs on a bounded interval;
    every verification gap fails closed.
    """

    def __init__(self, config: KeycloakConfig):
        config.validate()
        self._config = config
        self._opener = urllib.request.build_opener(_NoRedirectHandler())
        self._lock = threading.Lock()
        self._keys: dict[str, RSAPublicKey] = {}
        self._fetched_at = 0.0
        # Eager fetch: a misconfigured endpoint stops boot, not first request.
        self._refresh_keys()

    def _refresh_keys(self) -> None:
        request = urllib.request.Request(
            self._config.jwks_url, headers={"Accept": "application/json"}
        )
        try:
            with self._opener.open(request, timeout=JWKS_FETCH_TIMEOUT) as response:
                if response.status != 200:
                    raise AuthError(f"JWKS endpoint returned HTTP {response.status}")
                body = response.read(MAX_JWKS_BYTES + 1)
        except urllib.error.URLError as exc:
            raise AuthError(f"request JWKS: {exc}") from exc
        if len(body) > MAX_JWKS_BYTES:
            raise AuthError("JWKS document exceeds the size bound")
        try:
            document = json.loads(body)
        except json.JSONDecodeError as exc:
            raise AuthError(f"decode JWKS document: {exc}") from exc
        keys: dict[str, RSAPublicKey] = {}
        for jwk in document.get("keys", []):
            if jwk.get("kty") != "RSA" or not jwk.get("kid"):
                continue
            if jwk.get("use") not in (None, "sig"):
                continue
            try:
                modulus = int.from_bytes(_b64url_decode(jwk["n"]), "big")
                exponent = int.from_bytes(_b64url_decode(jwk["e"]), "big")
            except (KeyError, AuthError):
                continue
            if exponent < 3 or exponent % 2 == 0 or modulus.bit_length() < 2048:
                continue
            keys[jwk["kid"]] = RSAPublicNumbers(exponent, modulus).public_key()
        if not keys:
            raise AuthError("JWKS document contains no RSA signing keys")
        with self._lock:
            self._keys = keys
            self._fetched_at = time.monotonic()

    def _key_for(self, kid: str) -> RSAPublicKey:
        with self._lock:
            key = self._keys.get(kid)
            stale = time.monotonic() - self._fetched_at >= JWKS_MIN_REFRESH_INTERVAL
        if key is not None:
            return key
        if not stale:
            raise AuthError("unknown signing key")
        self._refresh_keys()
        with self._lock:
            key = self._keys.get(kid)
        if key is None:
            raise AuthError("unknown signing key")
        return key

    def authenticate(self, authorization: str | None) -> None:
        """Verify the Authorization header; raise AuthError on any gap."""
        if not authorization or not authorization.startswith("Bearer "):
            raise AuthError("bearer token is absent")
        token = authorization[len("Bearer "):].strip()
        if not token or len(token) > MAX_TOKEN_BYTES:
            raise AuthError("bearer token is absent or oversized")
        segments = token.split(".")
        if len(segments) != 3 or not all(segments):
            raise AuthError("token compact serialization is invalid")
        try:
            header = json.loads(_b64url_decode(segments[0]))
        except (json.JSONDecodeError, AuthError) as exc:
            raise AuthError("token header is invalid") from exc
        if header.get("alg") != "RS256" or not header.get("kid"):
            raise AuthError("token algorithm or key ID is invalid")
        key = self._key_for(str(header["kid"]))
        try:
            key.verify(
                _b64url_decode(segments[2]),
                f"{segments[0]}.{segments[1]}".encode("ascii"),
                padding.PKCS1v15(),
                hashes.SHA256(),
            )
        except InvalidSignature as exc:
            raise AuthError("token signature verification failed") from exc
        try:
            claims = json.loads(_b64url_decode(segments[1]))
        except (json.JSONDecodeError, AuthError) as exc:
            raise AuthError("token claims are invalid") from exc
        if claims.get("iss") != self._config.issuer:
            raise AuthError("token issuer is invalid")
        subject = claims.get("sub")
        if not isinstance(subject, str) or not subject.strip() or len(subject) > 256:
            raise AuthError("token subject is invalid")
        exp = claims.get("exp")
        if not isinstance(exp, (int, float)) or isinstance(exp, bool):
            raise AuthError("token has no valid expiry")
        now = time.time()
        if now >= exp:
            raise AuthError("token is expired")
        nbf = claims.get("nbf")
        if nbf is not None and (not isinstance(nbf, (int, float)) or now < nbf):
            raise AuthError("token is not yet valid")
        audience = claims.get("aud")
        audiences = set(audience if isinstance(audience, list) else [audience])
        azp = claims.get("azp")
        if isinstance(azp, str):
            audiences.add(azp)
        if self._config.audience not in audiences:
            raise AuthError("token is not issued for this API audience")


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: N805
        return None


def build_authenticator_from_env(env: dict | None = None) -> KeycloakAuthenticator | None:
    """Resolve the Keycloak authenticator from the environment.

    Returns None only on an explicitly non-production profile with no
    Keycloak coordinates set; every production or partial configuration gap
    raises BootError so the process refuses to start.
    """
    env = os.environ if env is None else env
    jwks_url = (env.get("KEYCLOAK_JWKS_URL") or "").strip()
    issuer = (env.get("KEYCLOAK_ISSUER") or "").strip()
    audience = (env.get("KEYCLOAK_EXPECTED_AUDIENCE") or "").strip()
    configured = [bool(v) for v in (jwks_url, issuer, audience)]
    if any(configured) and not all(configured):
        raise BootError(
            "KEYCLOAK_JWKS_URL, KEYCLOAK_ISSUER and KEYCLOAK_EXPECTED_AUDIENCE "
            "must be configured together"
        )
    if not all(configured):
        if production_profile(env):
            raise BootError(
                "production profile requires KEYCLOAK_JWKS_URL, KEYCLOAK_ISSUER "
                "and KEYCLOAK_EXPECTED_AUDIENCE for /score authentication"
            )
        log.warning(
            "scoring service running WITHOUT authentication — only valid on an "
            "explicit non-production profile (BEML_ENV=%s)",
            (env.get("BEML_ENV") or "").strip().lower(),
        )
        return None
    try:
        return KeycloakAuthenticator(
            KeycloakConfig(jwks_url=jwks_url, issuer=issuer, audience=audience)
        )
    except (AuthError, ValueError) as exc:
        raise BootError(f"keycloak authenticator init failed: {exc}") from exc


def require_auth(request: Request) -> None:
    """FastAPI dependency protecting /score (401 on any gap).

    A None authenticator is reachable only on an explicitly non-production
    profile (see build_authenticator_from_env).
    """
    authenticator: KeycloakAuthenticator | None = getattr(
        request.app.state, "authenticator", None
    )
    if authenticator is None:
        return
    try:
        authenticator.authenticate(request.headers.get("Authorization"))
    except AuthError as exc:
        raise HTTPException(
            status_code=401,
            detail="bearer token is absent or unverifiable",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
