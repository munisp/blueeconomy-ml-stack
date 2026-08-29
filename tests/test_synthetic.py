"""Synthetic generator contract tests: reproducible, labelled, structured."""

import numpy as np
import pandas as pd

from synthetic.ais import generate_ais
from synthetic.config import AisConfig, CvffConfig, DeclarationConfig
from synthetic.cvff import generate_cvff
from synthetic.declarations import generate_declarations, to_features


def test_declarations_reproducible():
    cfg = DeclarationConfig(n_declarations=500, n_consignees=100, seed=42)
    a, b = generate_declarations(cfg), generate_declarations(cfg)
    pd.testing.assert_frame_equal(a, b)


def test_declarations_labelled_synthetic_and_fraud_present():
    df = generate_declarations(DeclarationConfig(n_declarations=2000, seed=1))
    assert (df["data_source"] == "SYNTHETIC").all()
    assert 0.02 < df["is_fraud"].mean() < 0.15
    assert set(df["fraud_type"].unique()) >= {
        "none", "undervaluation", "hs_misclassification", "shell_consignee"}
    # undervalued rows really are cheap vs reference: factor 0.25-0.6 applied
    # to noisy honest prices (lognormal sigma 0.22), so we assert the
    # distribution is strongly depressed, not a hard per-row cap
    uv = df[df["fraud_type"] == "undervaluation"]
    assert uv["price_ratio_vs_reference"].mean() < 0.55
    assert (uv["price_ratio_vs_reference"] < 1.0).mean() > 0.9


def test_declarations_features_shape():
    df = generate_declarations(DeclarationConfig(n_declarations=300, seed=2))
    x, y = to_features(df)
    assert x.shape[0] == 300 and x.shape[1] == 11
    assert set(np.unique(y)) <= {0, 1}
    assert np.isfinite(x).all()


def test_ais_patterns_and_labels():
    df = generate_ais(AisConfig(n_vessels=30, days=5, seed=3))
    assert (df["data_source"] == "SYNTHETIC").all()
    labels = set(df["movement_label"].unique())
    assert "normal" in labels and labels != {"normal"}
    assert df["sog_kn"].between(0, 20).all()


def test_cvff_graph_labels():
    d = generate_cvff(CvffConfig(n_companies=300, n_accounts=400,
                                 n_transactions=2000, seed=4))
    cos, txs = d["companies"], d["transactions"]
    assert set(cos["label_name"].unique()) >= {"honest", "mule_shell", "roundtrip"}
    assert (cos["data_source"] == "SYNTHETIC").all()
    assert txs["amount_ngn"].gt(0).all()
