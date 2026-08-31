"""FastAPI scoring service template (CPU).

Fail-closed doctrine implemented end to end:
- /health reports per-model availability, never a liveness lie
- /score returns 200 with status=OK only when a real model produced a real
  number; otherwise 200 with status=SCORING_UNAVAILABLE and mode=rules_only
  so callers fall back to the deterministic rules engine
- /score NEVER fabricates a score

Run:  uvicorn inference.service:app --port 8100
Config via env: BEML_MODELS_ROOT (default ./models), BEML_AB_CONFIG
(default ./inference/ab_config.yaml), BEML_LATENCY_BUDGET_MS (default 50).
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from inference.events import build_publisher_from_env
from inference.scoring import STATUS_OK, Scorer

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
scorers = _build_scorers()
# Signed inference-event publisher (ml.inference.v1). None when
# BEML_EVENT_SINK=none; misconfiguration raises here and aborts boot.
publisher = build_publisher_from_env()


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


@app.post("/score/{model_key}")
def score(model_key: str, req: ScoreRequest) -> dict:
    scorer = scorers.get(model_key)
    if scorer is None:
        return {"status": "SCORING_UNAVAILABLE", "score": None, "mode": "rules_only",
                "detail": f"unknown model '{model_key}'"}
    result = scorer.score(req.features, entity_id=req.entity_id)
    payload = result.__dict__
    if result.status != STATUS_OK:
        # Explicit contract: caller MUST continue with deterministic rules only.
        payload["fallback"] = "deterministic_rules_only"
    if publisher is not None:
        # Publish a signed InferenceEvent (digests only, never raw features).
        # A publish failure does not alter the scoring response but is logged
        # loudly so operators can alert on the broken audit trail.
        try:
            publisher.publish_inference(
                model_name=result.model_name,
                model_version=result.model_version,
                status=result.status,
                score=result.score,
                mode=result.mode,
                latency_ms=result.latency_ms,
                entity_id=req.entity_id,
                features=req.features,
                detail=result.detail,
            )
        except Exception as exc:  # pragma: no cover - broker failure path
            import logging
            logging.getLogger(__name__).error(
                "ml.inference.v1 publish failed for %s: %s", model_key, exc
            )
    return payload
