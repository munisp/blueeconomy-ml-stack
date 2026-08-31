"""Signed inference events for the BlueEconomy ML stack (envelope v1.0).

Every scoring decision — OK or SCORING_UNAVAILABLE — is published to the
Kafka topic ``ml.inference.v1`` as a signed envelope (JWS-EdDSA over
RFC 8785 JCS, kid "<producer>-<epoch>"), implementing
``blueeconomy.ml.v1.InferenceEvent`` from blueeconomy-contracts.

Privacy contract: events carry the model name/version, the score, status,
latency, and SHA-256 digests of the entity ID and feature vector — NEVER the
raw features or raw entity identifiers (PII).

Fail-closed configuration:
- ``BEML_EVENT_SINK`` = ``none`` (default) | ``kafka`` | ``file``.
- When the sink is not ``none``, ``BEML_SIGNING_SEED_B64`` (32-byte Ed25519
  seed, base64url) is REQUIRED — the service refuses to start an unsigned
  publisher.
- ``kafka`` sink requires ``BEML_KAFKA_BROKERS``; ``file`` sink requires
  ``BEML_SPOOL_FILE`` (append-only JSONL spool, a real store-and-forward
  sink, not a mock).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from typing import Any

from inference.envelope import (
    SigningKey,
    build_envelope,
    sign_envelope,
)
from inference.jcs import canonicalize

log = logging.getLogger(__name__)

TOPIC_ML_INFERENCE = "ml.inference.v1"


class EventSink:
    def send(self, topic: str, envelope: dict[str, Any]) -> None:
        raise NotImplementedError

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


class KafkaEventSink(EventSink):
    def __init__(self, bootstrap_servers: str):
        from kafka import KafkaProducer

        self._producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers.split(","),
            value_serializer=lambda env: canonicalize(env),
            key_serializer=lambda k: k.encode("utf-8") if k else None,
            acks="all",
            retries=5,
        )

    def send(self, topic: str, envelope: dict[str, Any]) -> None:
        key = str(envelope.get("eventId", ""))
        future = self._producer.send(topic, key=key, value=envelope)
        future.get(timeout=30)  # surface broker failures

    def flush(self) -> None:
        self._producer.flush()

    def close(self) -> None:
        self._producer.close()


class FileEventSink(EventSink):
    """Append-only JSONL spool (store-and-forward)."""

    def __init__(self, spool_file: str):
        os.makedirs(os.path.dirname(os.path.abspath(spool_file)), exist_ok=True)
        self._fh = open(spool_file, "a", encoding="utf-8")
        self._lock = threading.Lock()

    def send(self, topic: str, envelope: dict[str, Any]) -> None:
        line = json.dumps({"topic": topic, "envelope": envelope}, separators=(",", ":"))
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()

    def flush(self) -> None:
        with self._lock:
            self._fh.flush()
            os.fsync(self._fh.fileno())

    def close(self) -> None:
        self.flush()
        self._fh.close()


class InferenceEventPublisher:
    """Signs and publishes blueeconomy.ml.v1.InferenceEvent envelopes."""

    def __init__(
        self,
        *,
        sink: EventSink,
        signing_key: SigningKey,
        producer: str,
        principal_id: str = "ml-stack-inference",
        principal_role: str = "SERVICE",
    ):
        self._sink = sink
        self._key = signing_key
        self._producer = producer
        self._principal_id = principal_id
        self._principal_role = principal_role

    def publish_inference(
        self,
        *,
        model_name: str,
        model_version: str | None,
        status: str,
        score: float | None,
        mode: str,
        latency_ms: float,
        entity_id: str,
        features: list[float],
        detail: str = "",
    ) -> dict[str, Any]:
        resource = {
            "modelName": model_name,
            "modelVersion": model_version or "",
            "status": status,
            "score": round(float(score), 6) if score is not None else None,
            "mode": mode,
            "latencyMs": round(float(latency_ms), 3),
            # Digests only — raw entity IDs / features never leave the service.
            "entityDigest": hashlib.sha256(entity_id.encode("utf-8")).hexdigest(),
            "inputDigest": hashlib.sha256(canonicalize(features)).hexdigest(),
            "detail": detail[:256],
        }
        envelope = build_envelope(
            event_type=TOPIC_ML_INFERENCE,
            resource_type="InferenceEvent",
            resource=resource,
            producer=self._producer,
            principal_id=self._principal_id,
            principal_role=self._principal_role,
            classification="INTERNAL",
        )
        signed = sign_envelope(envelope, self._key)
        self._sink.send(TOPIC_ML_INFERENCE, signed)
        return signed

    def flush(self) -> None:
        self._sink.flush()


def build_publisher_from_env() -> InferenceEventPublisher | None:
    """Build the publisher from env. Fail closed: a configured sink without a
    signing seed raises instead of publishing unsigned events."""
    sink_kind = os.environ.get("BEML_EVENT_SINK", "none").lower()
    if sink_kind == "none":
        return None

    producer = os.environ.get("BEML_PRODUCER", "ml-stack")
    epoch = int(os.environ.get("BEML_KEY_EPOCH", "0"))
    seed_b64 = os.environ.get("BEML_SIGNING_SEED_B64", "")
    if not seed_b64:
        raise RuntimeError(
            "SIGNING_KEY_REQUIRED: BEML_EVENT_SINK is configured but "
            "BEML_SIGNING_SEED_B64 is absent — refusing to publish unsigned "
            "inference events"
        )
    key = SigningKey.from_seed_b64(producer, epoch, seed_b64)

    if sink_kind == "kafka":
        brokers = os.environ.get("BEML_KAFKA_BROKERS", "")
        if not brokers:
            raise RuntimeError("BEML_KAFKA_BROKERS required for kafka sink")
        sink: EventSink = KafkaEventSink(brokers)
    elif sink_kind == "file":
        spool = os.environ.get("BEML_SPOOL_FILE", "")
        if not spool:
            raise RuntimeError("BEML_SPOOL_FILE required for file sink")
        sink = FileEventSink(spool)
    else:
        raise RuntimeError(f"unknown BEML_EVENT_SINK {sink_kind!r}")

    log.info("ml.inference.v1 publisher enabled (sink=%s, kid=%s)", sink_kind, key.kid)
    return InferenceEventPublisher(sink=sink, signing_key=key, producer=producer)
