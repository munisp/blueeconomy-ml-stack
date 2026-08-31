"""Tests for signed ml.inference.v1 inference events (envelope v1.0).

Proves:
1. Events are signed JWS-EdDSA/JCS envelopes that verify against the
   producer public key (kid "<producer>-<epoch>").
2. Events carry score/model-version/digests — never raw features or entity
   IDs (PII).
3. Fail-closed config: a configured sink without BEML_SIGNING_SEED_B64
   raises SIGNING_KEY_REQUIRED; unknown sink kinds raise.
4. SCORING_UNAVAILABLE decisions are also published (audit trail).
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from inference.envelope import SigningKey, verify_envelope, envelope_resource  # noqa: E402
from inference.events import (  # noqa: E402
    TOPIC_ML_INFERENCE,
    FileEventSink,
    InferenceEventPublisher,
    build_publisher_from_env,
)


class _CollectSink:
    def __init__(self):
        self.sent = []

    def send(self, topic, envelope):
        self.sent.append((topic, envelope))


@pytest.fixture()
def key():
    return SigningKey.generate("ml-stack", epoch=3)


@pytest.fixture()
def publisher(key):
    return InferenceEventPublisher(sink=_CollectSink(), signing_key=key, producer="ml-stack")


def _publish(pub):
    return pub.publish_inference(
        model_name="declaration-fraud",
        model_version="0.1.0",
        status="OK",
        score=0.87,
        mode="ml",
        latency_ms=3.2,
        entity_id="trader-12345",
        features=[1.0, 2.0, 3.0, 4.0],
    )


def test_signed_envelope_verifies(publisher, key):
    signed = _publish(publisher)
    directory = {key.kid: key.private_key.public_key()}
    kid = verify_envelope(signed, directory)
    assert kid == "ml-stack-3"
    header = signed["provenance"]["signature"].split(".")[0]
    import base64
    assert json.loads(base64.urlsafe_b64decode(header + "==")) == {
        "alg": "EdDSA",
        "kid": "ml-stack-3",
    }
    assert signed["envelopeVersion"] == "1.0"
    assert signed["eventType"] == TOPIC_ML_INFERENCE


def test_event_carries_digests_not_pii(publisher):
    signed = _publish(publisher)
    resource = envelope_resource(signed)
    assert resource["@type"] == "type.googleapis.com/blueeconomy.ml.v1.InferenceEvent"
    assert resource["modelName"] == "declaration-fraud"
    assert resource["modelVersion"] == "0.1.0"
    assert resource["score"] == pytest.approx(0.87)
    assert len(resource["entityDigest"]) == 64
    assert len(resource["inputDigest"]) == 64
    blob = json.dumps(signed)
    assert "trader-12345" not in blob  # raw entity ID never leaves the service
    assert "[1.0, 2.0, 3.0, 4.0]" not in blob  # raw features never leave
    assert "features" not in resource


def test_tampered_envelope_rejected(publisher, key):
    signed = _publish(publisher)
    signed["fhir"]["entry"][0]["resource"]["score"] = 0.01
    with pytest.raises(Exception):
        verify_envelope(signed, {key.kid: key.private_key.public_key()})


def test_unavailable_decisions_published(publisher):
    signed = publisher.publish_inference(
        model_name="vessel-anomaly",
        model_version="0.1.0",
        status="SCORING_UNAVAILABLE",
        score=None,
        mode="rules_only",
        latency_ms=1.1,
        entity_id="vessel-x",
        features=[0.0],
        detail="missing model file",
    )
    resource = envelope_resource(signed)
    assert resource["status"] == "SCORING_UNAVAILABLE"
    assert resource["score"] is None


def test_env_config_fail_closed_without_seed(monkeypatch):
    monkeypatch.setenv("BEML_EVENT_SINK", "file")
    monkeypatch.setenv("BEML_SPOOL_FILE", "/tmp/x.jsonl")
    monkeypatch.delenv("BEML_SIGNING_SEED_B64", raising=False)
    with pytest.raises(RuntimeError, match="SIGNING_KEY_REQUIRED"):
        build_publisher_from_env()


def test_env_config_unknown_sink(monkeypatch, key):
    monkeypatch.setenv("BEML_EVENT_SINK", "carrier-pigeon")
    monkeypatch.setenv("BEML_SIGNING_SEED_B64", key.seed_b64())
    with pytest.raises(RuntimeError, match="unknown BEML_EVENT_SINK"):
        build_publisher_from_env()


def test_env_none_disables(monkeypatch):
    monkeypatch.setenv("BEML_EVENT_SINK", "none")
    assert build_publisher_from_env() is None


def test_file_sink_spool_roundtrip(tmp_path, key):
    spool = tmp_path / "spool" / "events.jsonl"
    pub = InferenceEventPublisher(
        sink=FileEventSink(str(spool)), signing_key=key, producer="ml-stack"
    )
    pub.publish_inference(
        model_name="declaration-fraud", model_version="0.1.0", status="OK",
        score=0.5, mode="ml", latency_ms=2.0, entity_id="e", features=[1.0],
    )
    pub.flush()
    lines = spool.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["topic"] == TOPIC_ML_INFERENCE
    kid = verify_envelope(record["envelope"], {key.kid: key.private_key.public_key()})
    assert kid == key.kid


def test_service_publishes_on_score(monkeypatch, key):
    """End-to-end: /score emits a signed event to the configured sink."""
    monkeypatch.setenv("BEML_EVENT_SINK", "none")
    import inference.service as service

    importlib.reload(service)
    sink = _CollectSink()
    service.publisher = InferenceEventPublisher(
        sink=sink, signing_key=key, producer="ml-stack"
    )
    from fastapi.testclient import TestClient

    client = TestClient(service.app)
    resp = client.post(
        "/score/declaration-fraud",
        json={"entity_id": "ent-9", "features": [0.1] * 8},
    )
    assert resp.status_code == 200
    assert len(sink.sent) == 1
    topic, envelope = sink.sent[0]
    assert topic == TOPIC_ML_INFERENCE
    verify_envelope(envelope, {key.kid: key.private_key.public_key()})
    resource = envelope_resource(envelope)
    assert resource["modelName"] == "declaration-fraud"
    assert resource["status"] in ("OK", "SCORING_UNAVAILABLE")
