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
#    else results/runs.jsonl)
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

## Model coverage (Phase 17 audit outcomes)

- **Serving registry:** `declaration-fraud`, `vessel-anomaly` (#10 dark-vessel
  anomaly detection), `port-congestion` (G7, see below), and the Phase-18
  shadow-mode RL recommenders `berth-allocation`, `queue-policy`,
  `route-advice` are the only registered score routes.
- **Offline RL recommenders (Phase 18):** `training/rl/` trains
  `queue-policy` and `route-advice` as LinUCB contextual bandits (single-step
  decisions — no decision-caused state transition, so sequential RL would
  invent structure the data lacks) and `berth-allocation` as conservative
  CQL-H (allocation couples through berth occupancy, so it IS sequential;
  CQL's pessimism blocks extrapolation to unlogged actions; gamma=0 until a
  verified next-state join exists). Data is REAL logged rows only via
  config-gated DSNs (`BEML_RL_QUEUE_PG_DSN`, `BEML_RL_BERTH_PG_DSN`,
  `BEML_RL_ROUTE_PG_DSN`); no DSN or too little history exits honestly
  (`INSUFFICIENT_HISTORY`) and there is NO synthetic training default.
  Promotion is gated by off-policy evaluation (doubly-robust with IPS
  reported): a candidate that does not beat the logged baseline policy on
  the OPE estimate — or that strays off the logged support — is refused and
  nothing is exported. Serving is SHADOW-mode: responses carry
  `{"mode": "shadow", "policy_version", "action_index", "autonomous": false}`;
  recommendations are never auto-executed. Until an artifact passes the gate
  and is committed, all three routes report `SCORING_UNAVAILABLE`.
- **port-congestion (G7):** training path is `training/congestion.py`, fed
  ONLY by real `port_queue_observations` rows (geo-service migration 0013)
  via `BEML_CONGESTION_PG_DSN`. Until a real artifact is trained and
  committed under `models/port-congestion/`, the route honestly reports
  `SCORING_UNAVAILABLE` and geo-service answers 409 — no heuristic is ever
  served under this model name. Training exits `INSUFFICIENT_HISTORY`
  below the minimum supervised-pair count rather than fitting noise.
- **graph-mule-gnn (G6): deliberately excluded from serving.** The exported
  `models/graph-mule-gnn/0.1.0/model.onnx` bakes the frozen 6,721-node
  training context graph into the ONNX graph; executing it requires the full
  node-feature matrix, which the per-entity `/score` contract (one feature
  row per request) cannot supply — verified: single-row execution raises a
  Gather index-out-of-bounds inside the frozen context. It remains a
  training/eval artifact until a graph-context scoring API is built; it is
  not advertised as servable anywhere.
- **AEO / export credit scoring (#12): NO SUCH MODEL EXISTS and none is
  served.** `declaration-fraud` scores customs declaration fraud risk; it is
  NOT a creditworthiness model and must not be surfaced as one. Building an
  AEO/export credit score requires real credit/outcome labels that the
  platform does not yet collect; rather than ship a relabelled fraud model,
  no credit-score endpoint is exposed.

## License posture

All runtime dependencies are permissive-licensed (BSD/Apache-2.0/MIT);
see `requirements.txt` inline comments. Repository code: Apache-2.0.
