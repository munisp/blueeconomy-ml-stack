# blueeconomy-ml-stack

Real AI/ML/DL/GNN stack for the NewWave.io BlueEconomy PPP platform
(NPA, NIMASA, NIWA, FMMBE, CBN): actual PyTorch models, versioned trained
weights, MLflow-tracked training loops, lakehouse-fed continuous training,
Ray distributed compute, and fail-closed CPU inference.

## Architecture

```
                        +---------------------------+
                        |  blueeconomy-data-platform |
                        |  lakehouse (gold marts,    |
                        |  Parquet/GeoParquet)       |
                        +------------+--------------+
                                     |  (pipelines/extract.py; SYNTHETIC
                                     |   fallback below volume threshold)
                                     v
                        +---------------------------+
                        | versioned training         |
                        | snapshots (dataset_version |
                        | content hashes)            |
                        +------------+--------------+
                                     |
              +----------------------+----------------------+
              |                      |                      |
              v                      v                      v
   +-------------------+  +--------------------+  +--------------------+
   | training/tabular  |  | training/gnn       |  | training/anomaly   |
   | PyTorch MLP +     |  | GraphSAGE (PyG) on |  | autoencoder +      |
   | LightGBM baseline |  | trade/payment graph|  | IsolationForest    |
   +---------+---------+  +---------+----------+  +---------+----------+
             |  MLflow tracking (params/metrics/artifacts), seeds,
             |  early stopping, CPU default (--device cuda optional)
             v
   +-------------------+      evaluation gate      +--------------------+
   | models/<name>/    |  <--- beat deployed or -- | pipelines/         |
   | <semver>/model.*  |      never promote        | continuous_training|
   +---------+---------+                           +--------------------+
             | ONNX export (opset 17, parity-checked)
             v
   +-------------------+   deterministic hash A/B   +--------------------+
   | inference/service | <--- split on entity ID ---| monitoring/ab.py   |
   | FastAPI, CPU,     |                            +--------------------+
   | fail-closed       |   drift reports            +--------------------+
   +-------------------+ <--------------------------| monitoring/drift   |
             | SCORING_UNAVAILABLE -> rules-only    +--------------------+
             v
   platform deterministic rules engines (first line of defence, unchanged)

   ray_jobs/* : Ray Tune HPO + Ray Data prep; RAY_UNAVAILABLE -> local
```

## Quickstart (CPU only)

```bash
pip install -r requirements.txt            # pinned, permissive, CPU wheels
pip install -r requirements-optional.txt   # ray + evidently (optional)

# 1. generate SYNTHETIC training data (reproducible, seeded)
python -m synthetic.cli --out data/synthetic

# 2. train all three model families (MLflow if MLFLOW_TRACKING_URI set,
#    else results/runs.jsonl). PRODUCTION: MLFLOW_TRACKING_URI is mandatory
#    and must target the PostgreSQL-backed MLflow server — a file-based URI
#    (sqlite:///mlflow.db, file:..., bare path) refuses boot when BEML_ENV is
#    unset or "production". mlflow.db must never be committed (gitignored).
python -m training.tabular --version 0.1.0
python -m training.gnn     --version 0.1.0
python -m training.anomaly --version 0.1.0

# 3. export ONNX (parity-checked) and serve
python -m inference.export_onnx --model declaration-fraud --version 0.1.0
python -m inference.export_onnx --model vessel-anomaly     --version 0.1.0
uvicorn inference.service:app --port 8100

# 4. local MLflow + Ray head for dev
docker compose up mlflow ray-head

# 5. tests
pytest tests/ -q
```

## Continuous training from platform data

```bash
export BEML_LAKEHOUSE_ROOT=/path/to/lakehouse     # gold marts
python -m pipelines.extract                       # -> data/snapshots/<hash>/
python -m pipelines.continuous_training \
    --model declaration-fraud --candidate-version 0.2.0
```

The evaluation gate promotes only if the candidate beats the deployed model
on held-out AUROC by `--min-delta`. Promotion triggers ONNX export and a
registry-style `Staging -> Production` transition. When lakehouse volume is
below threshold and `BEML_ALLOW_SYNTHETIC_FALLBACK=1`, snapshots fall back to
clearly-labelled SYNTHETIC data (dev only — never set this in production).

## Honesty section — read this

What this repository IS:

- Real PyTorch/LightGBM/sklearn training loops with real learned weights,
  versioned under `models/<name>/<semver>/` with per-run `metrics.json`.
- Every metric in `models/MODEL_CARDS.md` comes from an actual training run
  executed in this workspace; raw run logs are in `results/`.
- Fail-closed inference: missing/invalid model, feature mismatch, or latency
  budget breach yields `SCORING_UNAVAILABLE` + rules-only mode. No score is
  ever fabricated.

What it is NOT:

- **Not validated on real fraud.** All training data is SYNTHETIC
  (statistically modelled, clearly labelled). The metrics measure the
  models' ability to learn the synthetic patterns, nothing more. Models must
  be retrained and re-validated on production data before any operational
  reliance; until then they are plumbing proofs, not fraud detectors.
- **Not a replacement for deterministic rules.** Rules engines remain the
  first line of defence; this stack augments them and degrades to
  rules-only when unavailable.
- Drift/A-B infrastructure is real but its dashboards/alerting wiring is a
  deploy-wave concern (see `deploy/README.md`).

## License posture

All runtime dependencies are permissive-licensed (BSD/Apache-2.0/MIT);
see `requirements.txt` inline comments. Repository code: Apache-2.0.
