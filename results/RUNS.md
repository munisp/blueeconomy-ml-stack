# Training runs — raw evidence

All weights committed under `models/` were produced by the runs below,
executed in this workspace (CPU-only, seeds as logged, MLflow tracking in
`mlruns/` — not committed). Machine-readable copies: `models/*/*/metrics.json`.

## Synthetic data generation

```
$ python -m synthetic.cli --out data/synthetic
[SYNTHETIC] wrote data/synthetic/declarations.parquet rows=20000 version=30c7274a7a2bf645
[SYNTHETIC] wrote data/synthetic/ais.parquet rows=120935 version=7a972b27328c9e66
[SYNTHETIC] wrote data/synthetic/cvff_companies.parquet rows=1800 version=f099fa2bdb539c78
[SYNTHETIC] wrote data/synthetic/cvff_accounts.parquet rows=2600 version=78946a03a5261cef
[SYNTHETIC] wrote data/synthetic/cvff_transactions.parquet rows=40000 version=9b346346fc6eb33b
```

## Run 1 — declaration-fraud-mlp 0.1.0 (+ LightGBM baseline)

```
$ python -m training.tabular --version 0.1.0 --device cpu
[early-stop] epoch=59 best_val_auroc=0.9649
[declaration-fraud-mlp] test: auroc=0.9708, auprc=0.8404, recall_at_precision_0.90=0.6605,
    best_f1=0.7696, best_f1_threshold=0.9626, temperature=0.9576
[declaration-fraud-lightgbm] test: auroc=0.9706, auprc=0.8239, recall_at_precision_0.90=0.6884,
    best_f1=0.7893, best_f1_threshold=0.6429
```

Splits: train 14,000 / val 3,000 / test 3,000 (stratified, seed 7).

## Run 2 — graph-mule-gnn 0.1.0 (PyTorch Geometric backend)

```
$ python -m training.gnn --version 0.1.0 --device cpu
[early-stop] epoch=151 best_val_auroc=0.9987
[graph-mule-gnn] backend=pyg test: auroc=0.9992, auprc=0.9945,
    recall_at_precision_0.90=0.9737, best_f1=0.9867, best_f1_threshold=0.6535
```

Graph: 1,800 companies / 2,600 accounts / 4,000 declarations / 120 vessels,
93,440 directed edges. NOTE: synthetic ring structure is strongly
separable — see model card limitations before reading anything into 0.999.

## Run 3 — vessel-anomaly-autoencoder 0.1.0 (+ IsolationForest baseline)

```
$ python -m training.anomaly --version 0.1.0 --device cpu
[early-stop] epoch=22 best_val_auroc=0.8585
[vessel-anomaly-autoencoder] test: auroc=0.8598, auprc=0.6427,
    recall_at_precision_0.90=0.4429, best_f1=0.6054
[vessel-anomaly-isoforest] test: auroc=0.9628, auprc=0.5404,
    recall_at_precision_0.90=0.1464, best_f1=0.5699
```

Honest note: the IsolationForest baseline beats the autoencoder on AUROC on
this synthetic data; the autoencoder wins recall at precision >= 0.90. Both
are shipped; model selection on production data is an open item.

## ONNX export verification (parity-checked, size budget <5MB)

```
[verify] models/declaration-fraud/0.1.0/model.onnx parity ok (7KB)
[verify] models/vessel-anomaly/0.1.0/model.onnx parity ok (3KB)
[verify] models/graph-mule-gnn/0.1.0/model.onnx parity ok (3903KB)
```

(The GNN artifact is larger because the training-graph adjacency is frozen
into the export for transductive CPU scoring; still within budget.)

## Ray Tune hyperparameter search (RAY_UNAVAILABLE -> local fallback)

```
$ python -m ray_jobs.tune_tabular --samples 4 --out results/tune_tabular.json
[RAY_UNAVAILABLE] Could not find any running Ray instance ...; falling back to local single-node
[local-sweep] {'hidden': 32, 'lr': 0.001, 'dropout': 0.1} -> val_auroc=0.8748
[local-sweep] {'hidden': 32, 'lr': 0.003, 'dropout': 0.1} -> val_auroc=0.9511
[local-sweep] {'hidden': 64, 'lr': 0.003, 'dropout': 0.1} -> val_auroc=0.9515
[local-sweep] {'hidden': 16, 'lr': 0.001, 'dropout': 0.1} -> val_auroc=0.8378
best: hidden=64, lr=0.003, dropout=0.1, val_auroc=0.9515
```

## Continuous training cycle (extract -> retrain -> gate -> promote -> export)

```
$ python -m pipelines.continuous_training --model declaration-fraud --candidate-version 0.1.1
[early-stop] epoch=59 best_val_auroc=0.9649
[declaration-fraud-mlp] test: auroc=0.9708 ... (same seed/config as 0.1.0)
gate: {"incumbent": null, "decision": "promote_first_model"}
[verify] models/declaration-fraud/0.1.1/model.onnx parity ok (7KB)
PRODUCTION pointer -> 0.1.1 ; notification appended to results/notifications.jsonl
```

## Drift report (scheduled-job artifact)

```
$ python -m monitoring.drift --reference data/synthetic/declarations.parquet \
    --current data/synthetic/declarations_current.parquet \
    --out results/drift/declaration_drift.json --prediction-col price_ratio_vs_reference
[drift] backend=evidently features=14 -> results/drift/declaration_drift.json
drifted features: [] (current period drawn from the same generator — honest negative)
```

## Inference service smoke (FastAPI TestClient)

```
GET  /health -> {"status": "ok", models: {declaration-fraud: 0.1.0 available, vessel-anomaly: 0.1.0 available}}
POST /score/declaration-fraud -> status=OK score=0.9339 latency_ms=0.61 (CPU)
POST /score/declaration-fraud (3 features) -> status=SCORING_UNAVAILABLE score=null
POST /score/unknown-model -> status=SCORING_UNAVAILABLE score=null
```
