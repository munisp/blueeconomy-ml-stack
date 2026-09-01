"""Signed inference-event publisher (Kafka topic ``ml.inference.v1``).

Every completed ``/score`` call produces exactly one InferenceEvent recording
the REAL outcome (OK with the real score, or SCORING_UNAVAILABLE with the
real reason — never a fabricated score). Events are envelope v1.0 signed
with JWS-EdDSA over the RFC 8785 JCS canonicalization, the same wire
contract as the other platform producers (see blueeconomy-contracts
docs/envelope-signature.md and blueeconomy-cv-service src/cvservice).

Fail-closed configuration (env only):
  BEML_INFERENCE_EVENTS_ENABLED   "true" to enable (default: disabled)
  BEML_KAFKA_BOOTSTRAP_SERVERS    required when enabled
  BEML_SIGNING_KEY_PATH           Ed25519 PKCS#8 PEM, required when enabled
  BEML_SIGNING_KEY_ID             JWS kid (default "blueeconomy-ml-stack-0")

When enabled but unconfigured, ``InferenceEventPublisher.from_env`` raises
EventConfigError at startup — the service never runs in a state where it
claims to publish but cannot. When disabled (default), ``from_env`` returns
None and /health honestly reports inference events as disabled.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import UTC, datetime
from typing import Any, Protocol

log = logging.getLogger("beml.inference.events")

TOPIC_INFERENCE = "ml.inference.v1"
EVENT_INFERENCE = "ml.inference.v1"
ENVELOPE_VERSION = "1.0"
PRODUCER = "blueeconomy-ml-stack"
RESOURCE_TYPE = "type.googleapis.com/blueeconomy.ml.v1.InferenceEvent"

DEFAULT_KEY_ID = "blueeconomy-ml-stack-0"


class EventConfigError(RuntimeError):
    """Fail-closed: events enabled but not honestly configurable."""

    CODE = "BEML_EVENTS_UNCONFIGURED"


# ── RFC 8785 JCS canonicalization ────────────────────────────────────────────
# Vendored clean-room implementation, byte-for-byte identical to
# blueeconomy-cv-service src/cvservice/crypto/jcs.py (same author, same
# contract): sorted keys (UTF-16 code-unit order), no whitespace, minimal
# string escaping, ECMAScript Number::toString for non-integers. Payloads
# here avoid non-integral numbers except score/latency, which take the
# unit-tested number path.


class CanonicalizationError(ValueError):
    pass


_ESCAPES = {
    '"': '\\"',
    "\\": "\\\\",
    "\b": "\\b",
    "\f": "\\f",
    "\n": "\\n",
    "\r": "\\r",
    "\t": "\\t",
}


def _escape_string(value: str) -> str:
    out: list[str] = ['"']
    for ch in value:
        esc = _ESCAPES.get(ch)
        if esc is not None:
            out.append(esc)
        elif ord(ch) < 0x20:
            out.append(f"\\u{ord(ch):04x}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def _shortest_digits(value: float) -> tuple[str, int]:
    rep = repr(abs(value))
    if "e" in rep or "E" in rep:
        mant, _, exp_s = rep.partition("e")
        e = int(exp_s)
    else:
        mant, e = rep, 0
    if "." in mant:
        int_part, _, frac_part = mant.partition(".")
    else:
        int_part, frac_part = mant, ""
    digits = (int_part + frac_part).lstrip("0")
    exp10 = e - len(frac_part)
    trailing = len(digits) - len(digits.rstrip("0"))
    digits = digits.rstrip("0") or "0"
    exp10 += trailing
    return digits, exp10


def _number_to_string(value: float) -> str:
    if value != value or value in (float("inf"), float("-inf")):
        raise CanonicalizationError("JCS cannot represent NaN or Infinity")
    if value == 0:
        return "0"
    sign = "-" if value < 0 else ""
    mantissa_digits, exp10 = _shortest_digits(value)
    n = len(mantissa_digits) + exp10
    k = len(mantissa_digits)
    if k <= n <= 21:
        return sign + mantissa_digits + "0" * (n - k)
    if 0 < n <= 21:
        return sign + mantissa_digits[:n] + "." + mantissa_digits[n:]
    if -6 < n <= 0:
        return sign + "0." + "0" * (-n) + mantissa_digits
    if k == 1:
        mant = mantissa_digits
    else:
        mant = mantissa_digits[0] + "." + mantissa_digits[1:]
    exp = n - 1
    exp_sign = "+" if exp >= 0 else ""
    return f"{sign}{mant}e{exp_sign}{exp}"


def _encode(value: Any, out: list[str]) -> None:
    if value is None:
        out.append("null")
    elif value is True:
        out.append("true")
    elif value is False:
        out.append("false")
    elif isinstance(value, str):
        out.append(_escape_string(value))
    elif isinstance(value, int):
        if abs(value) > 9007199254740991:
            raise CanonicalizationError(f"integer {value} exceeds IEEE-754 safe range")
        out.append(str(value))
    elif isinstance(value, float):
        if value.is_integer() and abs(value) <= 9007199254740991:
            out.append(str(int(value)))
        else:
            out.append(_number_to_string(value))
    elif isinstance(value, (list, tuple)):
        out.append("[")
        for i, item in enumerate(value):
            if i:
                out.append(",")
            _encode(item, out)
        out.append("]")
    elif isinstance(value, dict):
        items = sorted(value.items(), key=lambda kv: kv[0].encode("utf-16-be", "surrogatepass"))
        out.append("{")
        for i, (k, v) in enumerate(items):
            if i:
                out.append(",")
            if not isinstance(k, str):
                raise CanonicalizationError("object keys must be strings")
            out.append(_escape_string(k))
            out.append(":")
            _encode(v, out)
        out.append("}")
    else:
        raise CanonicalizationError(f"unsupported type for JCS: {type(value).__name__}")


def canonicalize_bytes(value: Any) -> bytes:
    out: list[str] = []
    _encode(value, out)
    return "".join(out).encode("utf-8")


# ── JWS-EdDSA ────────────────────────────────────────────────────────────────


def _b64u(data: bytes) -> str:
    import base64

    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def jws_sign(private_key: Any, kid: str, payload: bytes) -> str:
    """Compact JWS (EdDSA) over ``payload`` — same construction as
    blueeconomy-cv-service crypto/eddsa.jws_sign."""
    header = _b64u(canonicalize_bytes({"alg": "EdDSA", "kid": kid}))
    body = _b64u(payload)
    signing_input = f"{header}.{body}".encode("ascii")
    signature = private_key.sign(signing_input)
    return f"{header}.{body}.{_b64u(signature)}"


def load_private_key(path: str) -> Any:
    from cryptography.hazmat.primitives.serialization import load_pem_private_key

    with open(path, "rb") as fh:
        return load_pem_private_key(fh.read(), None)


# ── Envelope v1.0 ────────────────────────────────────────────────────────────


def build_inference_envelope(
    *,
    entity_id: str,
    model_name: str,
    model_version: str | None,
    status: str,
    score: float | None,
    mode: str,
    latency_ms: float,
    detail: str | None = None,
    key_id: str = DEFAULT_KEY_ID,
    event_id: str | None = None,
) -> dict[str, Any]:
    """Unsigned envelope v1.0 around the InferenceEvent resource. Carries the
    real scoring outcome — score is null (never invented) when unavailable."""
    event_id = event_id or f"evt-{uuid.uuid4()}"
    resource: dict[str, Any] = {
        "@type": RESOURCE_TYPE,
        "entityId": entity_id,
        "modelName": model_name,
        "modelVersion": model_version,
        "status": status,
        "score": score,
        "mode": mode,
        "latencyMs": round(float(latency_ms), 3),
    }
    if detail:
        resource["detail"] = detail
    return {
        "envelopeVersion": ENVELOPE_VERSION,
        "eventId": event_id,
        "eventType": EVENT_INFERENCE,
        "occurredAt": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "producer": PRODUCER,
        "correlationId": event_id,
        "classification": "RESTRICTED",
        "fhir": {
            "resourceType": "Bundle",
            "type": "message",
            "bundleId": f"bdl-{uuid.uuid4()}",
            "entry": [{"fullUrl": f"urn:uuid:{uuid.uuid4()}", "resource": resource}],
        },
        "provenance": {
            "principalId": PRODUCER,
            "principalRole": "ml-inference",
            "ledgerCommitHash": "",
            "signature": "",
            "keyId": key_id,
        },
    }


def sign_envelope(envelope: dict[str, Any], private_key: Any, kid: str) -> dict[str, Any]:
    prov = {k: v for k, v in envelope["provenance"].items() if k != "signature"}
    payload = canonicalize_bytes({**envelope, "provenance": prov})
    return {
        **envelope,
        "provenance": {**envelope["provenance"], "signature": jws_sign(private_key, kid, payload)},
    }


# ── Publisher ────────────────────────────────────────────────────────────────


class Producer(Protocol):
    def send(self, topic: str, value: bytes) -> Any: ...

    def flush(self) -> Any: ...


class InferenceEventPublisher:
    """Publishes one signed InferenceEvent per completed score call."""

    def __init__(self, producer: Producer, private_key: Any, key_id: str = DEFAULT_KEY_ID) -> None:
        self._producer = producer
        self._key = private_key
        self._kid = key_id

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "InferenceEventPublisher | None":
        """Fail-closed factory. Returns None when events are disabled
        (default); raises EventConfigError when enabled but unconfigured."""
        e = os.environ if env is None else env
        if (e.get("BEML_INFERENCE_EVENTS_ENABLED", "false")).lower() != "true":
            return None
        bootstrap = e.get("BEML_KAFKA_BOOTSTRAP_SERVERS", "")
        key_path = e.get("BEML_SIGNING_KEY_PATH", "")
        if not bootstrap:
            raise EventConfigError(
                f"{EventConfigError.CODE}: BEML_KAFKA_BOOTSTRAP_SERVERS is required "
                "when BEML_INFERENCE_EVENTS_ENABLED=true"
            )
        if not key_path:
            raise EventConfigError(
                f"{EventConfigError.CODE}: BEML_SIGNING_KEY_PATH is required "
                "when BEML_INFERENCE_EVENTS_ENABLED=true"
            )
        from kafka import KafkaProducer

        producer: Producer = KafkaProducer(
            bootstrap_servers=bootstrap.split(","),
            acks="all",
            enable_idempotence=True,
        )
        return cls(
            producer=producer,
            private_key=load_private_key(key_path),
            key_id=e.get("BEML_SIGNING_KEY_ID", DEFAULT_KEY_ID),
        )

    def publish(
        self,
        *,
        entity_id: str,
        model_name: str,
        model_version: str | None,
        status: str,
        score: float | None,
        mode: str,
        latency_ms: float,
        detail: str | None = None,
    ) -> dict[str, Any]:
        """Build, sign and send one InferenceEvent. Returns the signed
        envelope (tests introspect it). A Kafka failure is logged as an
        error — the score result itself was already honestly computed and is
        never altered to hide an event-pipeline problem."""
        envelope = build_inference_envelope(
            entity_id=entity_id,
            model_name=model_name,
            model_version=model_version,
            status=status,
            score=score,
            mode=mode,
            latency_ms=latency_ms,
            detail=detail,
            key_id=self._kid,
        )
        signed = sign_envelope(envelope, self._key, self._kid)
        try:
            self._producer.send(TOPIC_INFERENCE, json.dumps(signed).encode("utf-8"))
            self._producer.flush()
        except Exception as exc:  # honest logging, never silent
            log.error("failed to publish InferenceEvent for %s: %s", entity_id, exc)
        return signed
