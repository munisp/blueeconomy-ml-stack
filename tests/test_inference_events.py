"""Signed InferenceEvent publisher tests (phase 11 remediation:
ml.inference.v1 was a documented claim with zero publishers — this module
is the real publisher, wired into inference/service.py after each score).

Covers: fail-closed env configuration, envelope v1.0 shape, JWS-EdDSA
signature over the RFC 8785 JCS canonical payload (verified with the real
public key), honest score=null on SCORING_UNAVAILABLE, and exactly one
event per score call.
"""

from __future__ import annotations

import base64
import json

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from inference.events import (
    DEFAULT_KEY_ID,
    EVENT_INFERENCE,
    TOPIC_INFERENCE,
    EventConfigError,
    InferenceEventPublisher,
    build_inference_envelope,
    canonicalize_bytes,
    sign_envelope,
)


class FakeProducer:
    def __init__(self) -> None:
        self.sent: list[tuple[str, bytes]] = []
        self.flushed = 0

    def send(self, topic: str, value: bytes):
        self.sent.append((topic, value))

    def flush(self):
        self.flushed += 1


def _key() -> Ed25519PrivateKey:
    return Ed25519PrivateKey.generate()


def _verify(signed: dict, public_key) -> None:
    """Full JWS-EdDSA verification of provenance.signature against the JCS
    canonicalization of the envelope minus the signature."""
    jws = signed["provenance"]["signature"]
    header_b64, body_b64, sig_b64 = jws.split(".")
    header = json.loads(base64.urlsafe_b64decode(header_b64 + "=="))
    assert header == {"alg": "EdDSA", "kid": DEFAULT_KEY_ID}
    prov = {k: v for k, v in signed["provenance"].items() if k != "signature"}
    expected_body = base64.urlsafe_b64encode(
        canonicalize_bytes({**signed, "provenance": prov})
    ).rstrip(b"=").decode("ascii")
    assert body_b64 == expected_body  # payload is the JCS canonical bytes
    sig = base64.urlsafe_b64decode(sig_b64 + "==")
    public_key.verify(sig, f"{header_b64}.{body_b64}".encode("ascii"))  # raises if bad


def test_from_env_disabled_by_default():
    assert InferenceEventPublisher.from_env({}) is None
    assert InferenceEventPublisher.from_env({"BEML_INFERENCE_EVENTS_ENABLED": "false"}) is None


def test_from_env_enabled_requires_kafka_and_key():
    with pytest.raises(EventConfigError, match="BEML_KAFKA_BOOTSTRAP_SERVERS"):
        InferenceEventPublisher.from_env({"BEML_INFERENCE_EVENTS_ENABLED": "true"})
    with pytest.raises(EventConfigError, match="BEML_SIGNING_KEY_PATH"):
        InferenceEventPublisher.from_env({
            "BEML_INFERENCE_EVENTS_ENABLED": "true",
            "BEML_KAFKA_BOOTSTRAP_SERVERS": "kafka:9092",
        })


def test_envelope_shape_and_honest_fields():
    env = build_inference_envelope(
        entity_id="decl-1", model_name="declaration-fraud", model_version="0.1.0",
        status="OK", score=0.42, mode="ml", latency_ms=3.21,
    )
    assert env["envelopeVersion"] == "1.0"
    assert env["eventType"] == EVENT_INFERENCE == "ml.inference.v1"
    assert env["producer"] == "blueeconomy-ml-stack"
    resource = env["fhir"]["entry"][0]["resource"]
    assert resource["@type"] == "type.googleapis.com/blueeconomy.ml.v1.InferenceEvent"
    assert resource["entityId"] == "decl-1"
    assert resource["score"] == 0.42
    assert resource["modelVersion"] == "0.1.0"


def test_unavailable_event_has_null_score_never_fabricated():
    env = build_inference_envelope(
        entity_id="decl-2", model_name="declaration-fraud", model_version="0.1.0",
        status="SCORING_UNAVAILABLE", score=None, mode="rules_only",
        latency_ms=1.0, detail="feature count 3 != model expects 11",
    )
    resource = env["fhir"]["entry"][0]["resource"]
    assert resource["score"] is None
    assert resource["status"] == "SCORING_UNAVAILABLE"
    assert "feature count" in resource["detail"]


def test_publisher_sends_signed_envelope_to_ml_inference_topic():
    key = _key()
    producer = FakeProducer()
    pub = InferenceEventPublisher(producer=producer, private_key=key)
    signed = pub.publish(
        entity_id="decl-1", model_name="declaration-fraud", model_version="0.1.0",
        status="OK", score=0.9, mode="ml", latency_ms=2.0,
    )
    assert len(producer.sent) == 1
    topic, payload = producer.sent[0]
    assert topic == TOPIC_INFERENCE
    assert producer.flushed == 1
    wire = json.loads(payload)
    assert wire["eventId"] == signed["eventId"]
    _verify(wire, key.public_key())


def test_tampered_envelope_fails_verification():
    key = _key()
    producer = FakeProducer()
    pub = InferenceEventPublisher(producer=producer, private_key=key)
    pub.publish(
        entity_id="decl-1", model_name="declaration-fraud", model_version="0.1.0",
        status="OK", score=0.9, mode="ml", latency_ms=2.0,
    )
    wire = json.loads(producer.sent[0][1])
    wire["fhir"]["entry"][0]["resource"]["score"] = 0.01  # tamper
    with pytest.raises(Exception):
        _verify(wire, key.public_key())


def test_publish_failure_logged_not_hidden(caplog):
    class BoomProducer:
        def send(self, topic, value):
            raise RuntimeError("broker down")

        def flush(self):
            pass

    pub = InferenceEventPublisher(producer=BoomProducer(), private_key=_key())
    with caplog.at_level("ERROR"):
        signed = pub.publish(
            entity_id="decl-1", model_name="m", model_version="1",
            status="OK", score=0.1, mode="ml", latency_ms=1.0,
        )
    assert signed["provenance"]["signature"]  # envelope still honestly built
    assert "broker down" in caplog.text
