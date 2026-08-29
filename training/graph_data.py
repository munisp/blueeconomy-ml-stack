"""Build the heterogeneous trade/payment graph for GNN training.

Node types: company, account, declaration, vessel.
Edge types: company-owns-account, company-transacts-company,
company-files-declaration, company-operates-vessel.

Linkage note (honesty): synthetic declarations and AIS vessels use independent
ID spaces; declaration->company and vessel->company links are assigned by a
documented deterministic hash so the GNN has real structure to learn over.
The company node label (mule/shell vs honest) comes from the SYNTHETIC CVFF
generator ground truth and is NEVER used as a feature.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

NODE_TYPES = ["company", "account", "declaration", "vessel"]


def _hash_link(key: str, n: int) -> int:
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % n


def build_graph(data_dir: str | Path, max_declarations: int = 4000,
                seed: int = 7) -> dict:
    """Return node features, edge_index, node-type ids and company labels.

    The graph is materialised as a homogeneous projection (node_type one-hot
    prepended to features) so it trains identically with the PyG backend and
    the pure-torch GraphSAGE fallback.
    """
    data_dir = Path(data_dir)
    companies = pd.read_parquet(data_dir / "cvff_companies.parquet")
    accounts = pd.read_parquet(data_dir / "cvff_accounts.parquet")
    txs = pd.read_parquet(data_dir / "cvff_transactions.parquet")
    decl = pd.read_parquet(data_dir / "declarations.parquet")
    ais = pd.read_parquet(data_dir / "ais.parquet")

    rng = np.random.default_rng(seed)
    n_co, n_ac = len(companies), len(accounts)
    decl = decl.sample(min(max_declarations, len(decl)), random_state=seed).reset_index(drop=True)
    n_de = len(decl)
    vessels = ais.groupby("imo").agg(
        mean_sog=("sog_kn", "mean"),
        frac_anom=("movement_label", lambda s: float((s != "normal").mean())),
    ).reset_index()
    n_ve = len(vessels)

    # --- node features (padded to a common width, node-type one-hot first) ---
    def block(rows: np.ndarray, type_id: int) -> np.ndarray:
        onehot = np.zeros((len(rows), len(NODE_TYPES)), dtype=np.float32)
        onehot[:, type_id] = 1.0
        return np.hstack([onehot, rows.astype(np.float32)])

    co_tx_out = txs.groupby("from_company")["amount_ngn"].agg(["count", "sum"])
    co_tx_in = txs.groupby("to_company")["amount_ngn"].agg(["count", "sum"])
    co_feat = pd.DataFrame({"company_id": companies["company_id"]})
    co_feat = co_feat.merge(co_tx_out, left_on="company_id", right_index=True, how="left") \
                     .rename(columns={"count": "out_deg", "sum": "out_amt"})
    co_feat = co_feat.merge(co_tx_in, left_on="company_id", right_index=True, how="left",
                            suffixes=("", "_in")).rename(
                                columns={"count": "in_deg", "sum": "in_amt"})
    co_feat[["out_deg", "out_amt", "in_deg", "in_amt"]] = \
        co_feat[["out_deg", "out_amt", "in_deg", "in_amt"]].fillna(0)
    co_rows = np.column_stack([
        np.log1p(companies["annual_turnover_ngn"]),
        np.log1p(co_feat["out_deg"]), np.log1p(co_feat["in_deg"]),
        np.log1p(co_feat["out_amt"]), np.log1p(co_feat["in_amt"]),
    ])

    ac_feat = accounts.merge(
        txs.groupby("from_account")["amount_ngn"].agg(["count", "sum"]),
        left_on="account_id", right_index=True, how="left").fillna(0)
    ac_rows = np.column_stack([
        ac_feat["bank_code"].astype(float) / 100.0,
        np.log1p(ac_feat["count"]), np.log1p(ac_feat["sum"]),
        np.zeros((n_ac, 2)),
    ])

    de_rows = np.column_stack([
        np.log1p(decl["cif_value_usd"]), decl["price_ratio_vs_reference"],
        decl["duty_rate_applied"], decl["consignee_is_shell"].astype(float),
        np.zeros(n_de),
    ])

    ve_rows = np.column_stack([
        vessels["mean_sog"], vessels["frac_anom"],
        np.zeros((n_ve, 3)),
    ])

    x = np.vstack([
        block(co_rows, 0), block(ac_rows, 1), block(de_rows, 2), block(ve_rows, 3)])

    # --- edges (global node ids offset by type block) ---
    off_ac, off_de, off_ve = n_co, n_co + n_ac, n_co + n_ac + n_de
    acct_gid = {a: off_ac + i for i, a in enumerate(accounts["account_id"])}
    comp_gid = {c: i for i, c in enumerate(companies["company_id"])}

    e_own = np.array([[comp_gid[c], acct_gid[a]] for c, a in
                      zip(accounts["company_id"], accounts["account_id"])]).T
    e_tx = np.array([[comp_gid[s], comp_gid[d]] for s, d in
                     zip(txs["from_company"], txs["to_company"])]).T
    e_de = np.array([[off_de + i, _hash_link(str(t), n_co)]
                     for i, t in enumerate(decl["consignee_tin"])]).T
    e_ve = np.array([[off_ve + i, _hash_link(f"imo-{imo}", n_co)]
                     for i, imo in enumerate(vessels["imo"])]).T
    edges = np.hstack([e_own, e_own[::-1], e_tx, e_tx[::-1], e_de, e_de[::-1],
                       e_ve, e_ve[::-1]])

    y = (companies["node_label"] > 0).to_numpy(dtype=np.int64)  # mule/roundtrip = 1
    return {
        "x": x, "edge_index": edges.astype(np.int64), "y": y,
        "n_company": n_co, "node_feature_dim": x.shape[1],
        "meta": {"companies": n_co, "accounts": n_ac, "declarations": n_de,
                 "vessels": n_ve, "edges": edges.shape[1]},
    }
