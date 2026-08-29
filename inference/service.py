"""FastAPI scoring service template (CPU).

Fail-closed doctrine implemented end to end:
- /health reports per-model availability, never a liveness lie
- /score requires a verified Keycloak RS256 bearer token (production
  profile refuses to boot without KEYCLOAK_JWKS_URL / KEYCLOAK_ISSUER /
  KEYCLOAK_EXPECTED_AUDIENCE); /health stays public
- /score returns 200 with status=OK only when a real model produced a real
  number; otherwise 200 with status=SCORING_UNAVAILABLE and mode=rules_only
  so callers fall back to the deterministic rules engine
- /score NEVER fabricates a score
- the production profile (BEML_ENV unset or "production") refuses a
  file-based MLFLOW_TRACKING_URI (sqlite/file/bare path)

Run:  uvicorn inference.service:app --port 8100
Config via env: BEML_MODELS_ROOT (default ./models), BEML_AB_CONFIG
(default ./inference/ab_config.yaml), BEML_LATENCY_BUDGET_MS (default 50),
BEML_ENV, KEYCLOAK_JWKS_URL / KEYCLOAK_ISSUER / KEYCLOAK_EXPECTED_AUDIENCE,
MLFLOW_TRACKING_URI.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from inference.auth import build_authenticator_from_env, require_auth
from inference.scoring import STATUS_OK, Scorer
from training.common import validate_tracking_uri

MODELS_ROOT = Path(os.environ.get("BEML_MODELS_ROOT", "models"))
AB_CONFIG = os.environ.get("BEML_AB_CONFIG", "inference/ab_config.yaml")
LATENCY_BUDGET_MS = float(os.environ.get("BEML_LATENCY_BUDGET_MS", "50"))

# model registry: name -> candidate versions (A/B split from ab_config when present)
MODEL_REGISTRY = {
    "declaration-fraud": {"model_name": "declaration-fraud", "versions": ["0.1.0"]},
    "vessel-anomaly": {"model_name": "vessel-anomaly", "versions": ["0.1.0"]},
}


def _build_scorers() -> dict[str, Scorer]:
    scorers = {}
    ab = None
    if Path(AB_CONFIG).is_file():
        try:
            from monitoring.ab import HashSplitter
            ab = AB_CONFIG
        except Exception:
            ab = None
    for key, spec in MODEL_REGISTRY.items():
        versions, weights = spec["versions"], None
        if ab:
            try:
                from monitoring.ab import HashSplitter
                splitter = HashSplitter.from_config(ab, spec["model_name"])
                versions, weights = splitter.versions, splitter.weights
            except Exception:
                pass  # unknown model in config -> registry default, fail-closed later
        scorers[key] = Scorer(MODELS_ROOT, spec["model_name"], versions,
                              split=weights, latency_budget_ms=LATENCY_BUDGET_MS)
    return scorers


app = FastAPI(title="BlueEconomy ML Scoring (fail-closed, CPU)", version="0.1.0")
# OTel (Phase-7): no-op unless OTEL_EXPORTER_OTLP_ENDPOINT is set — the
# sanctioned fail-open; scoring never depends on telemetry.
from inference.telemetry import init_telemetry

init_telemetry(app, service_name="blueeconomy-ml-stack", version="0.1.0")
# Boot gates (fail closed): Keycloak coordinates mandatory in production;
# a file-based MLflow tracking URI is refused in production. The inference
# path records no runs, so the URI itself is not required here.
app.state.authenticator = build_authenticator_from_env()
validate_tracking_uri(required_in_production=False)
scorers = _build_scorers()


class ScoreRequest(BaseModel):
    entity_id: str = Field(..., description="Stable entity ID for A/B routing")
    features: list[float]


@app.get("/health")
def health() -> dict:
    report = {}
    for key, scorer in scorers.items():
        availability = {}
        for v in scorer.versions:
            availability[v] = "available" if scorer._load(v) is not None else "unavailable"
        report[key] = availability
    degraded = any(s == "unavailable" for rep in report.values() for s in rep.values())
    return {"status": "degraded" if degraded else "ok", "models": report,
            "doctrine": "deterministic rules first; ML augments; fail-closed"}


@app.post("/score/{model_key}", dependencies=[Depends(require_auth)])
def score(model_key: str, req: ScoreRequest) -> dict:
    scorer = scorers.get(model_key)
    if scorer is None:
        return {"status": "SCORING_UNAVAILABLE", "score": None, "mode": "rules_only",
                "detail": f"unknown model '{model_key}'"}
    # Per-model scoring span (Phase-7 OTel; no-op when telemetry disabled).
    from inference.telemetry import get_tracer

    with get_tracer("beml.inference").start_as_current_span(
        "ml.score", attributes={"ml.model": model_key}
    ) as span:
        result = scorer.score(req.features, entity_id=req.entity_id)
        span.set_attribute("ml.model_version", result.model_version or "")
        span.set_attribute("ml.score_status", result.status)
        span.set_attribute("ml.latency_ms", round(result.latency_ms, 3))
    payload = result.__dict__
    if result.status != STATUS_OK:
        # Explicit contract: caller MUST continue with deterministic rules only.
        payload["fallback"] = "deterministic_rules_only"
    return payload
