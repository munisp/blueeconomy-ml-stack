"""Training smoke tests: real loops, tiny data, few epochs — metrics must be
better than chance (proves the loops learn, without paying full-train cost
in CI)."""

from pathlib import Path

import pandas as pd
import pytest

from synthetic.ais import generate_ais
from synthetic.cli import dataset_version
from synthetic.config import AisConfig, CvffConfig, DeclarationConfig
from synthetic.cvff import generate_cvff
from synthetic.declarations import generate_declarations


@pytest.fixture(scope="module")
def synth_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("synthetic")
    generate_declarations(DeclarationConfig(n_declarations=4000, seed=11)).to_parquet(
        out / "declarations.parquet", index=False)
    generate_ais(AisConfig(n_vessels=40, days=7, seed=12)).to_parquet(
        out / "ais.parquet", index=False)
    cvff = generate_cvff(CvffConfig(n_companies=600, n_accounts=800,
                                    n_transactions=6000, seed=13))
    cvff["companies"].to_parquet(out / "cvff_companies.parquet", index=False)
    cvff["accounts"].to_parquet(out / "cvff_accounts.parquet", index=False)
    cvff["transactions"].to_parquet(out / "cvff_transactions.parquet", index=False)
    return out


def test_tabular_learns(synth_dir, tmp_path):
    from training.tabular import build_argparser, train
    args = build_argparser().parse_args([
        "--data", str(synth_dir / "declarations.parquet"),
        "--out", str(tmp_path / "m"), "--version", "t", "--epochs", "30"])
    metrics = train(args)
    assert metrics["test"]["auroc"] > 0.7
    assert (tmp_path / "m" / "t" / "model.pt").is_file()


def test_gnn_learns(synth_dir, tmp_path):
    from training.gnn import build_argparser, train
    args = build_argparser().parse_args([
        "--data-dir", str(synth_dir), "--out", str(tmp_path / "g"),
        "--version", "t", "--epochs", "60"])
    metrics = train(args)
    assert metrics["test"]["auroc"] > 0.6


def test_anomaly_learns(synth_dir, tmp_path):
    from training.anomaly import build_argparser, train
    args = build_argparser().parse_args([
        "--data", str(synth_dir / "ais.parquet"),
        "--out", str(tmp_path / "a"), "--version", "t", "--epochs", "40"])
    metrics = train(args)
    assert metrics["test"]["auroc"] > 0.6


def test_dataset_version_stable(synth_dir):
    df = pd.read_parquet(synth_dir / "declarations.parquet")
    assert dataset_version(df) == dataset_version(df.copy())
