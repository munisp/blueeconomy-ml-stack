# Deploy notes — k8s/helm requirements for the gitops wave

This file DOCUMENTS requirements; `blueeconomy-platform-gitops` owns the
actual charts/manifests. Nothing here is a live manifest.

## Workloads to add

1. **scoring-service** (Deployment, 2+ replicas, CPU-only)
   - Image built from repo root: `pip install -r requirements.txt` then
     `uvicorn inference.service:app --port 8100`.
   - Env: `BEML_MODELS_ROOT`, `BEML_AB_CONFIG`, `BEML_LATENCY_BUDGET_MS`,
     `MLFLOW_TRACKING_URI`.
   - Resources: requests 250m CPU / 512Mi, limits 2 CPU / 1Gi (models are
     <5MB; the budget is for feature assembly + onnxruntime).
   - Models mounted read-only from the model artifact store (see below).
   - HPA on CPU is fine; scale-to-zero is NOT (cold start would breach the
     fail-closed latency contract).

2. **mlflow** (Deployment + Service) with:
   - Postgres backend store (reuse the platform Postgres or a dedicated
     CloudNativePG cluster); connection via Secret, never inline.
   - Artifact root on the lakehouse object storage (same env-driven
     BLUEECONOMY_STORAGE_* convention as blueeconomy-data-platform).

3. **ray-cluster** (KubeRay `RayCluster`): head + N CPU worker groups for
   `ray_jobs/*`. Only needed for scheduled retraining windows; size to the
   tuning budget. Ray dashboard must NOT be exposed publicly.

4. **CronJobs**:
   - `snapshot-builder`: `python -m pipelines.extract` (daily).
   - `continuous-training`: `python -m pipelines.continuous_training ...`
     (daily/weekly per model). Must run with `BEML_ALLOW_SYNTHETIC_FALLBACK`
     unset in production — synthetic fallback is a dev convenience only.
   - `drift-report`: `python -m monitoring.drift ...` (daily) writing
     artifacts to the lakehouse monitoring prefix.

## Model artifact distribution

- `pipelines.continuous_training` promotes by registry stage transition and
  exports ONNX. The gitops wave should sync `models/<name>/PRODUCTION`
  artifacts into a read-only volume (or object-store sync sidecar) consumed
  by the scoring-service.
- Version routing config (`inference/ab_config.yaml`) is a ConfigMap;
  changes are rollout-safe because bucketing is deterministic by entity ID.

## Security / doctrine requirements

- Scoring-service responses with `status=SCORING_UNAVAILABLE` must be
  alerting-visible (SLO: availability of ML scoring; deterministic rules
  remain the first line and are never blocked by this service).
- No GPU nodes required anywhere in the default path.
- SYNTHETIC data must never reach production namespaces: the synthetic
  fallback env flag must be absent from production CronJob specs.
