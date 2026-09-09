"""OpenTelemetry setup - exports traces to the cluster's shared collector
(otel-collector.otel.svc.cluster.local), visible in Jaeger/Grafana.

Guarded by OTEL_EXPORTER_OTLP_ENDPOINT: unset means telemetry is fully
disabled and boot proceeds exactly as before - the same fail-open contract
every other service on this platform uses.
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger("blueeco-ml-stack")


def init_telemetry(app) -> None:
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    if not endpoint:
        log.info("telemetry: OTEL_EXPORTER_OTLP_ENDPOINT unset, tracing disabled")
        return

    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    resource = Resource.create({
        "service.name": os.environ.get("OTEL_SERVICE_NAME", "blueeco-ml-stack"),
        "service.namespace": "blueeconomy",
        "deployment.environment": os.environ.get("OTEL_ENVIRONMENT", "local"),
    })
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app, excluded_urls="health")
    log.info("telemetry: OTLP export enabled -> %s", endpoint)
