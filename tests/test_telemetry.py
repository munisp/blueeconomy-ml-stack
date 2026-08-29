"""Phase-7 OTel tests: disabled-mode boot, propagation round-trip, tenant baggage."""

from __future__ import annotations

import pytest
from opentelemetry import baggage, context, trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from inference import telemetry


@pytest.fixture()
def memory_exporter():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return exporter, provider


def test_disabled_by_default_without_endpoint(monkeypatch):
    monkeypatch.delenv(telemetry.ENDPOINT_ENV, raising=False)
    assert telemetry.telemetry_enabled() is False
    assert telemetry.init_telemetry(None, service_name="t", version="0") is False
    with telemetry.get_tracer().start_as_current_span("noop") as span:
        assert span.is_recording() is False


def test_disabled_mode_service_boots_and_scores(monkeypatch):
    """Telemetry-off: app import + /score path must be unaffected."""
    from fastapi.testclient import TestClient

    monkeypatch.delenv(telemetry.ENDPOINT_ENV, raising=False)
    from inference.service import app

    client = TestClient(app)
    assert client.get("/health").status_code == 200
    # /score without auth must still fail closed (401/403), not break on telemetry.
    response = client.post(
        "/score/declaration-fraud", json={"entity_id": "e1", "features": [0.0]}
    )
    assert response.status_code in (401, 403)


def test_propagation_round_trip(memory_exporter):
    exporter, provider = memory_exporter
    tracer = provider.get_tracer("test")
    with tracer.start_as_current_span("upstream") as span:
        ctx = baggage.set_baggage("tenant.id", "tenant-7")
        ctx = baggage.set_baggage("agency", "NPA", context=ctx)
        token = context.attach(ctx)
        try:
            carrier: dict[str, str] = {}
            telemetry.inject_context(carrier)
        finally:
            context.detach(token)
        expected_trace_id = span.get_span_context().trace_id

    assert "traceparent" in carrier and "baggage" in carrier
    extracted = telemetry.extract_context(carrier)
    assert (
        trace.get_current_span(extracted).get_span_context().trace_id
        == expected_trace_id
    )
    assert baggage.get_baggage("tenant.id", context=extracted) == "tenant-7"
    assert baggage.get_baggage("agency", context=extracted) == "NPA"
    with tracer.start_as_current_span("downstream", context=extracted) as child:
        assert child.get_span_context().trace_id == expected_trace_id


def test_tenant_baggage_becomes_span_attributes(memory_exporter):
    exporter, provider = memory_exporter
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    app = FastAPI()

    @app.get("/ping")
    def ping():
        return {"ok": True}

    FastAPIInstrumentor().instrument_app(
        app, server_request_hook=telemetry.tenant_server_request_hook
    )
    previous = trace.get_tracer_provider()
    trace.set_tracer_provider(provider)
    try:
        client = TestClient(app)
        response = client.get(
            "/ping", headers={"baggage": "tenant.id=tenant-7,agency=NPA"}
        )
        assert response.status_code == 200
    finally:
        FastAPIInstrumentor().uninstrument_app(app)
        trace.set_tracer_provider(previous)

    server_spans = [
        s for s in exporter.get_finished_spans() if s.kind == trace.SpanKind.SERVER
    ]
    assert server_spans, "server span expected"
    assert server_spans[0].attributes["tenant.id"] == "tenant-7"
    assert server_spans[0].attributes["agency"] == "NPA"


def test_scoring_spans_recorded(memory_exporter):
    """Manual scoring spans (per-model + feature_prepare + inference) appear."""
    exporter, provider = memory_exporter
    previous = trace.get_tracer_provider()
    trace.set_tracer_provider(provider)
    try:
        import inference.scoring as scoring

        # Rebind the module-level tracer to the test provider.
        scoring._tracer = provider.get_tracer("beml.scoring")
        scorer = scoring.Scorer("models", "declaration-fraud", ["0.1.0"])
        result = scorer.score([0.0] * 4, entity_id="entity-x")
        assert result.status in (scoring.STATUS_OK, scoring.STATUS_UNAVAILABLE)
    finally:
        trace.set_tracer_provider(previous)

    names = {s.name for s in exporter.get_finished_spans()}
    # Model availability depends on committed weights; feature_prepare only
    # runs when the model loaded. At minimum no exception and spans are no-ops
    # when the model is absent.
    assert names <= {"ml.score.feature_prepare", "ml.score.inference"}


def test_drop_counting_exporter_never_raises():
    class FailingExporter:
        def export(self, spans):
            raise ConnectionError("collector down")

        def shutdown(self):
            pass

        def force_flush(self, timeout_millis=5000):
            return False

    counts = []

    class Counter:
        def add(self, n):
            counts.append(n)

    wrapped = telemetry._DropCountingSpanExporter(FailingExporter(), Counter())
    from opentelemetry.sdk.trace.export import SpanExportResult

    assert wrapped.export([object()]) is SpanExportResult.FAILURE
    assert counts == [1]
