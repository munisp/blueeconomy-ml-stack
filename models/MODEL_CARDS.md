# Model cards

All metrics below were produced by actual training runs executed in this
workspace (see `results/RUNS.md` for raw output; per-model `metrics.json`
next to each artifact). **All training data is SYNTHETIC.** None of these
models has ever seen, let alone been validated on, a real fraud case.

License for all model artifacts: Apache-2.0. Weights are products of the
training runs documented in `results/RUNS.md`.

**Run-backing guarantee:** every `models/<name>/<semver>/` directory in this
registry has a corresponding real training-run entry in `results/runs.jsonl`
with a matching metrics hash, enforced by `python -m pipelines.registry_check`
(regression test: `tests/test_registry.py`). A version without a logged run
is not a release.

---

## declaration-fraud-mlp — 0.1.0 (PRODUCTION)

> **Registry correction (ML-1):** a `0.1.1` version was briefly "promoted"
> on 2026-08-29, but it was a same-seed duplicate of 0.1.0 (byte-identical
> artifacts, zero metric improvement, no logged training run) that slipped
> through a gate shortcut when no PRODUCTION pointer existed. The 0.1.1
> promotion has been rescinded: its artifacts are removed (retained in git
> history), `PRODUCTION` points at 0.1.0, and an append-only
> `REJECTED_DUPLICATE` correction is recorded in
> `results/notifications.jsonl`. Full evidence: `results/RUNS.md`.
> 0.1.0 is the only valid declaration-fraud release.


- **Architecture:** PyTorch MLP, 11 features -> 32 -> 32 -> 1, dropout 0.15,
  post-hoc temperature calibration (T=0.9576). ~2.6k parameters. Artifacts:
  `model.pt` (9KB), `model.onnx` (7KB, opset 17, parity-verified),
  `baseline_lightgbm.joblib` (1.0MB).
- **Intended use:** augment the deterministic customs rules engine with an
  undervaluation / HS-misclassification / shell-consignee risk score for
  Nigerian import declarations.
- **Training data:** SYNTHETIC (`synthetic/declarations.py`, seed 20240,
  dataset_version `30c7274a7a2bf645`, 20,000 rows, 7% fraud).
  Splits train/val/test = 14,000/3,000/3,000, stratified, seed 7.
- **Metrics (held-out test, from the actual run):**
  AUROC **0.9708**, AUPRC **0.8404**, recall@precision>=0.90 **0.6605**,
  best-F1 **0.7696**.
  LightGBM baseline (same split): AUROC 0.9706, AUPRC 0.8239,
  recall@p>=0.90 0.6884, best-F1 0.7893 — i.e. the MLP is NOT better than
  the gradient-boosting baseline on this data; both are shipped honestly.
- **Limitations:** synthetic-only; real declaration fraud is adaptive and
  adversarial in ways this generator does not model. Never validated on real
  fraud cases — synthetic-only until production data flows. Must be
  retrained and re-gated via `pipelines/continuous_training.py` before
  operational reliance.

## graph-mule-gnn — 0.1.0

- **Architecture:** GraphSAGE (PyTorch Geometric `SAGEConv`, 2 layers,
  hidden 24, mean aggregation) over the heterogeneous trade/payment graph
  (companies, accounts, declarations, vessels; 93,440 directed edges),
  binary head for mule/shell node classification. Artifacts: `model.pt`
  (10KB), `model.onnx` (3.9MB — training graph adjacency frozen in for
  transductive CPU scoring; inductive scoring on new graphs must use
  `model.pt`).
- **Intended use:** candidate shell-company / mule-account detection in the
  CVFF contribution/payment graph, augmenting deterministic graph rules.
- **Training data:** SYNTHETIC CVFF graph (seeds 20240-20242; 1,800 company
  nodes, labels from generator ground truth; 60/20/20 node split, seed 7).
- **Metrics (held-out test nodes, from the actual run):**
  AUROC **0.9992**, AUPRC **0.9945**, recall@precision>=0.90 **0.9737**.
- **Limitations:** the near-perfect score reflects that synthetic mule rings
  are structurally regular (fixed ring size, narrow amount band) — real
  rings will not be. Treat as plumbing proof only. Declaration->company and
  vessel->company links are deterministic hash assignments (documented in
  `training/graph_data.py`), a synthetic simplification. Never validated on
  real fraud cases.

## vessel-anomaly-autoencoder — 0.1.0

- **Architecture:** PyTorch autoencoder 6 -> 16 -> 4 -> 16 -> 6, MSE
  reconstruction error as anomaly score, trained semi-supervised on normal
  AIS pings only. Artifacts: `model.pt` (5KB), `model.onnx` (3KB,
  parity-verified), `baseline_isoforest.joblib` (3.6MB).
- **Intended use:** flag anomalous vessel movement (loitering, dark-gap
  reappearance, EEZ incursion) augmenting geofence rules.
- **Training data:** SYNTHETIC AIS (`synthetic/ais.py`, 120 vessels, 21
  days, 120,935 pings, dataset_version `7a972b27328c9e66`).
- **Metrics (held-out test, from the actual run):**
  autoencoder AUROC **0.8598**, AUPRC **0.6427**, recall@precision>=0.90
  **0.4429**.
  IsolationForest baseline: AUROC **0.9628**, AUPRC 0.5404,
  recall@p>=0.90 0.1464. **The baseline beats the autoencoder on AUROC**;
  the autoencoder wins recall at high precision. We ship both and say so.
- **Limitations:** synthetic-only; real AIS noise (GPS jitter, satellite
  vs terrestrial coverage, class-B transceivers) is not modelled. Never
  validated on real movement anomalies.

---

## Fail-closed contract (all models)

If a model artifact is missing or invalid, or a request is malformed, or the
CPU latency budget (default 50ms) is exceeded, the inference service returns
`SCORING_UNAVAILABLE` and the caller must proceed with deterministic rules
only. No score is ever fabricated.
