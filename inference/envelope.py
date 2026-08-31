"""Canonical signed event envelopes for the BlueEconomy platform.

Implements the fleet signature scheme byte-for-byte as specified in
``blueeconomy-contracts/docs/envelope-signature.md``:

- ``provenance.signature`` is a JWS compact serialization (RFC 7515) using
  EdDSA over Ed25519 (RFC 8037), three non-empty base64url (no padding)
  segments.
- Protected header is exactly ``{"alg":"EdDSA","kid":"<producer>-<epoch>"}``.
- Payload is the JCS-canonicalized (RFC 8785) JSON of the full envelope
  excluding ``provenance.signature``, UTF-8 encoded.
- The Ed25519 signature is computed over
  ``base64url(header) + "." + base64url(payload)``.

Envelope shape mirrors ``blueeconomy.contracts.v1.EventEnvelope``
(envelopeVersion ``1.0``) with the FHIR message-Bundle wire form used by the
``geo.*.v1`` fixtures; ML event resources carry an ``@type`` of
``type.googleapis.com/blueeconomy.ml.v1.<Message>``.
"""

from __future__ import annotations

import base64
import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PrivateFormat,
    PublicFormat,
    NoEncryption,
)

from .jcs import canonicalize

ENVELOPE_VERSION = "1.0"
ML_TYPE_PREFIX = "type.googleapis.com/blueeconomy.ml.v1."

_KID_RE = re.compile(r"^[A-Za-z0-9._-]{1,256}$")


def b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def b64url_decode(segment: str) -> bytes:
    if not segment or not re.fullmatch(r"[A-Za-z0-9_-]+", segment):
        raise ValueError("not base64url (no padding permitted)")
    pad = "=" * (-len(segment) % 4)
    return base64.urlsafe_b64decode(segment + pad)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


class SignatureError(ValueError):
    """Envelope signature failure, with a spec reason code."""

    def __init__(self, reason_code: str, detail: str = ""):
        self.reason_code = reason_code
        super().__init__(f"{reason_code}: {detail}" if detail else reason_code)


@dataclass(frozen=True)
class SigningKey:
    """Producer Ed25519 signing key and its directory kid."""

    kid: str
    private_key: Ed25519PrivateKey

    @classmethod
    def generate(cls, producer: str, epoch: int = 0) -> "SigningKey":
        return cls(
            kid=f"{producer}-{epoch}",
            private_key=Ed25519PrivateKey.generate(),
        )

    @classmethod
    def from_seed_b64(cls, producer: str, epoch: int, seed_b64: str) -> "SigningKey":
        seed = b64url_decode(seed_b64)
        if len(seed) != 32:
            raise ValueError("Ed25519 seed must be 32 bytes")
        return cls(
            kid=f"{producer}-{epoch}",
            private_key=Ed25519PrivateKey.from_private_bytes(seed),
        )

    def seed_b64(self) -> str:
        return b64url_encode(
            self.private_key.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
        )

    def public_key_b64(self) -> str:
        return b64url_encode(
            self.private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        )


def build_envelope(
    *,
    event_type: str,
    resource_type: str,
    resource: dict[str, Any],
    producer: str,
    principal_id: str,
    principal_role: str,
    classification: str = "INTERNAL",
    correlation_id: str | None = None,
    event_id: str | None = None,
    occurred_at: str | None = None,
) -> dict[str, Any]:
    """Build an unsigned envelope (``provenance.signature`` empty)."""
    return {
        "envelopeVersion": ENVELOPE_VERSION,
        "eventId": event_id or f"evt-{uuid.uuid4()}",
        "eventType": event_type,
        "occurredAt": occurred_at or utc_now_iso(),
        "producer": producer,
        "correlationId": correlation_id or f"corr-{uuid.uuid4()}",
        "fhir": {
            "resourceType": "Bundle",
            "type": "message",
            "bundleId": f"bdl-{uuid.uuid4()}",
            "entry": [
                {
                    "fullUrl": f"urn:uuid:{uuid.uuid4()}",
                    "resource": {
                        "@type": ML_TYPE_PREFIX + resource_type,
                        **resource,
                    },
                }
            ],
        },
        "provenance": {
            "principalId": principal_id,
            "principalRole": principal_role,
            "ledgerCommitHash": "",
            "signature": "",
        },
        "classification": classification,
    }


def envelope_resource(envelope: dict[str, Any]) -> dict[str, Any]:
    """Extract the primary event resource; fail closed on malformed shape."""
    entry = envelope["fhir"]["entry"]
    if not isinstance(entry, list) or len(entry) != 1:
        raise ValueError("envelope fhir.entry must contain exactly one resource")
    return entry[0]["resource"]


def _payload_bytes(envelope: dict[str, Any]) -> bytes:
    """JCS of the full envelope excluding ``provenance.signature``."""
    provenance = dict(envelope.get("provenance", {}))
    provenance.pop("signature", None)
    stripped = {**envelope, "provenance": provenance}
    return canonicalize(stripped)


def sign_envelope(envelope: dict[str, Any], key: SigningKey) -> dict[str, Any]:
    """Return a copy of ``envelope`` with ``provenance.signature`` populated."""
    header = json.dumps({"alg": "EdDSA", "kid": key.kid}, separators=(",", ":")).encode()
    seg_header = b64url_encode(header)
    seg_payload = b64url_encode(_payload_bytes(envelope))
    signing_input = f"{seg_header}.{seg_payload}".encode("ascii")
    signature = key.private_key.sign(signing_input)
    signed = json.loads(json.dumps(envelope))  # deep copy
    signed["provenance"]["signature"] = (
        f"{seg_header}.{seg_payload}.{b64url_encode(signature)}"
    )
    return signed


def load_key_directory(path: str) -> dict[str, Ed25519PublicKey]:
    """Load the mounted producer public-key directory (fail closed)."""
    import os

    if not os.path.isfile(path) or os.path.islink(path):
        raise SignatureError("key-directory", "absent, unreadable, or not a regular file")
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict) or not raw:
        raise SignatureError("key-directory", "not a non-empty JSON object")
    directory: dict[str, Ed25519PublicKey] = {}
    for kid, value in raw.items():
        if not _KID_RE.match(kid) or not isinstance(value, str):
            raise SignatureError("key-directory", f"malformed entry for {kid!r}")
        try:
            key_bytes = b64url_decode(value)
        except ValueError as exc:
            raise SignatureError("key-directory", f"malformed key for {kid!r}") from exc
        if len(key_bytes) != 32:
            raise SignatureError("key-directory", f"malformed key for {kid!r}")
        directory[kid] = Ed25519PublicKey.from_public_bytes(key_bytes)
    return directory


def verify_envelope(
    envelope: dict[str, Any],
    key_directory: dict[str, Ed25519PublicKey],
) -> str:
    """Fail-closed verification per envelope-signature.md §4.

    Returns the ``kid`` on success; raises :class:`SignatureError` with the
    spec reason code on any rejection. Rejection is terminal.
    """
    signature = envelope.get("provenance", {}).get("signature")
    if not isinstance(signature, str):
        raise SignatureError("malformed-jws", "provenance.signature missing")
    segments = signature.split(".")
    if len(segments) != 3 or any(not s for s in segments):
        raise SignatureError("malformed-jws", "expected three non-empty segments")
    seg_header, seg_payload, seg_signature = segments
    try:
        header = json.loads(b64url_decode(seg_header))
    except (ValueError, json.JSONDecodeError) as exc:
        raise SignatureError("malformed-jws", "undecodable protected header") from exc
    if header.get("alg") != "EdDSA":
        raise SignatureError("unsupported-alg", f"alg={header.get('alg')!r}")
    kid = header.get("kid")
    if not isinstance(kid, str) or not _KID_RE.match(kid):
        raise SignatureError("malformed-jws", "malformed kid")
    public_key = key_directory.get(kid)
    if public_key is None:
        raise SignatureError("unknown-kid", kid)
    try:
        payload = b64url_decode(seg_payload)
    except ValueError as exc:
        raise SignatureError("malformed-jws", "undecodable payload") from exc
    if payload != _payload_bytes(envelope):
        raise SignatureError("payload-mismatch")
    try:
        public_key.verify(
            b64url_decode(seg_signature),
            f"{seg_header}.{seg_payload}".encode("ascii"),
        )
    except Exception as exc:
        raise SignatureError("invalid-signature") from exc
    return kid
