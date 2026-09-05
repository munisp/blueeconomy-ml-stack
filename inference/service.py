"""FastAPI scoring service template (CPU).

Fail-closed doctrine implemented end to end:
- /health reports per-model availability, never a liveness lie
- /score returns 200 with status=OK only when a real model produced a real
  number; otherwise 200 with status=SCORING_UNAVAILABLE and mode=rules_only
  so callers fall back to the deterministic rules engine
- /score NEVER fabricates a score
- /score, /docs and /openapi.json require a verified Keycloak bearer token
  (blueeconomy realm); when OIDC is not configured they return 503 —
  there is no fail-open anonymous path. /health stays public (probe).
- Request bodies are capped (BEML_MAX_BODY_BYTES, default 256 KiB) with a
  clean 413, and every response carries HTTP security headers.

Run:  uvicorn inference.service:app --port 8100
Config via env: BEML_MODELS_ROOT (default ./models), BEML_AB_CONFIG
(default ./inference/ab_config.yaml), BEML_LATENCY_BUDGET_MS (default 50),
BEML_MAX_BODY_BYTES (default 262144), BEML_OIDC_JWKS_PATH /
BEML_OIDC_JWKS_URL / BEML_OIDC_ISSUER / BEML_OIDC_AUDIENCE.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from inference.auth import AuthError, AuthSettings, Identity, JwksKeyring, verify_bearer
from inference.scoring import STATUS_OK, Scorer

MODELS_ROOT = Path(os.environ.get("BEML_MODELS_ROOT", "models"))
AB_CONFIG = os.environ.get("BEML_AB_CONFIG", "inference/ab_config.yaml")
LATENCY_BUDGET_MS = float(os.environ.get("BEML_LATENCY_BUDGET_MS", "50"))
MAX_BODY_BYTES = int(os.environ.get("BEML_MAX_BODY_BYTES", str(256 * 1024)))

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


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # OIDC optional; when configured the JWKS must load or boot fails.
    app.state.auth_settings = AuthSettings.from_env()
    app.state.keyring = (
        JwksKeyring.load(app.state.auth_settings)
        if app.state.auth_settings.oidc_configured
        else None
    )
    yield


app = FastAPI(
    title="BlueEconomy ML Scoring (fail-closed, CPU)",
    version="0.1.0",
    lifespan=lifespan,
    # API schema/docs are gated behind bearer auth below (they disclose the
    # scoring contract); there are no unauthenticated documentation routes.
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
scorers = _build_scorers()


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


@app.middleware("http")
async def limit_request_body(request: Request, call_next):
    """Reject oversized request bodies with a clean 413 (DoS guard)."""
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared = int(content_length)
        except ValueError:
            return JSONResponse(status_code=400, content={"detail": "invalid Content-Length"})
        if declared > MAX_BODY_BYTES:
            return _too_large()
    body = await request.body()
    if len(body) > MAX_BODY_BYTES:
        return _too_large()
    return await call_next(request)


def _too_large() -> JSONResponse:
    return JSONResponse(
        status_code=413,
        content={"detail": "request body too large", "limit_bytes": MAX_BODY_BYTES},
    )


async def require_identity(request: Request) -> Identity:
    """Verified OIDC identity or 401/503. Fail-closed: when OIDC is not
    configured there is no anonymous fallback — 503."""
    settings: AuthSettings = getattr(request.app.state, "auth_settings", AuthSettings())
    keyring = getattr(request.app.state, "keyring", None)
    if not settings.oidc_configured or keyring is None:
        raise HTTPException(
            status_code=503,
            detail={"reason": "auth-oidc-unavailable",
                    "detail": "OIDC is not configured; authenticated routes fail closed"},
        )
    auth = request.headers.get("authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail={"reason": "missing-bearer"})
    try:
        return verify_bearer(auth.removeprefix("Bearer ").strip(), keyring, settings)
    except AuthError as exc:
        raise HTTPException(status_code=401, detail={"reason": exc.reason, "detail": str(exc)}) from exc


def _build_event_publisher():
    """Signed InferenceEvent publisher (topic ml.inference.v1). Disabled by
    default; when BEML_INFERENCE_EVENTS_ENABLED=true the configuration must
    be complete or startup fails closed (EventConfigError)."""
    from inference.events import EventConfigError, InferenceEventPublisher

    try:
        return InferenceEventPublisher.from_env()
    except EventConfigError:
        raise  # fail closed: never run claiming to publish while unable
    except Exception as exc:
        raise EventConfigError(
            f"{EventConfigError.CODE}: publisher init failed: {exc}"
        ) from exc


event_publisher = _build_event_publisher()


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
            "inference_events": "enabled" if event_publisher is not None else "disabled",
            "doctrine": "deterministic rules first; ML augments; fail-closed"}


@app.post("/score/{model_key}")
def score(model_key: str, req: ScoreRequest, identity: Identity = Depends(require_identity)) -> dict:
    scorer = scorers.get(model_key)
    if scorer is None:
        return {"status": "SCORING_UNAVAILABLE", "score": None, "mode": "rules_only",
                "detail": f"unknown model '{model_key}'"}
    result = scorer.score(req.features, entity_id=req.entity_id)
    payload = result.__dict__
    if result.status != STATUS_OK:
        # Explicit contract: caller MUST continue with deterministic rules only.
        payload["fallback"] = "deterministic_rules_only"
    if event_publisher is not None:
        # One signed InferenceEvent per completed score call, recording the
        # REAL outcome (score may honestly be None on SCORING_UNAVAILABLE).
        event_publisher.publish(
            entity_id=req.entity_id,
            model_name=result.model_name,
            model_version=result.model_version,
            status=result.status,
            score=result.score,
            mode=result.mode,
            latency_ms=result.latency_ms,
            detail=result.detail,
        )
    return payload


@app.get("/openapi.json", include_in_schema=False)
async def gated_openapi(identity: Identity = Depends(require_identity)) -> JSONResponse:
    return JSONResponse(app.openapi())


@app.get("/docs", include_in_schema=False)
async def gated_docs(identity: Identity = Depends(require_identity)):
    return get_swagger_ui_html(openapi_url="/openapi.json", title=f"{app.title} - Swagger UI")
