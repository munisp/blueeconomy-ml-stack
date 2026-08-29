"""SYNTHETIC CVFF (Cabotage Vessel Financing Fund) payment/graph generator.

Builds a company/account/transaction population with embedded financial-crime
patterns and ground-truth node labels for the GNN:

- mule rings: clusters of accounts cycling contribution-like payments
- round-tripping: funds leaving and returning to inflate contribution history
- insider collusion: companies sharing an "officer" node behave coordinately

Produces both a transaction table and the heterogeneous graph tables
(companies, accounts, edges) consumed by training/gnn.py.

DATA IS SYNTHETIC. No real TINs, accounts or persons.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CvffConfig


def generate_cvff(cfg: CvffConfig = CvffConfig()) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(cfg.seed)

    companies = pd.DataFrame({
        "company_id": [f"CO-{i:06d}" for i in range(cfg.n_companies)],
        "annual_turnover_ngn": rng.lognormal(21.0, 1.1, cfg.n_companies).round(2),
        "officer_id": [f"OFF-{rng.integers(0, int(cfg.n_companies * 0.6)):05d}"
                       for _ in range(cfg.n_companies)],
    })
    accounts = pd.DataFrame({
        "account_id": [f"AC-{i:06d}" for i in range(cfg.n_accounts)],
        "company_id": rng.choice(companies["company_id"], cfg.n_accounts),
        "bank_code": rng.choice(["044", "058", "033", "011", "070"], cfg.n_accounts),
    })

    # --- label companies: mules, round-trippers, honest ---
    labels = np.zeros(cfg.n_companies, dtype=np.int64)  # 0 honest, 1 mule/shell, 2 roundtrip
    mule_members: list[list[int]] = []
    chosen = rng.choice(cfg.n_companies, cfg.n_mule_rings * cfg.mule_ring_size, replace=False)
    for r in range(cfg.n_mule_rings):
        ring = chosen[r * cfg.mule_ring_size:(r + 1) * cfg.mule_ring_size]
        mule_members.append(list(ring))
        labels[ring] = 1
    remaining = np.where(labels == 0)[0]
    n_rt = int(cfg.roundtrip_rate * cfg.n_companies)
    rt = rng.choice(remaining, n_rt, replace=False)
    labels[rt] = 2
    companies["node_label"] = labels  # GNN target
    companies["label_name"] = pd.Series(labels).map({0: "honest", 1: "mule_shell", 2: "roundtrip"})

    acct_by_company = accounts.groupby("company_id")["account_id"].apply(list).to_dict()

    all_accounts = accounts["account_id"].to_numpy()

    def acct_of(cid: str) -> str:
        owned = acct_by_company.get(cid)
        if not owned:  # company drew no account: route via a random account
            return str(all_accounts[rng.integers(0, len(all_accounts))])
        return owned[rng.integers(0, len(owned))]

    # --- transactions ---
    rows = []
    tx_types = ["cvff_contribution", "loan_drawdown", "repayment", "transfer"]
    for i in range(cfg.n_transactions):
        kind = rng.random()
        if kind < 0.06 and len(mule_members):
            # mule ring cycling: chain around the ring
            ring = mule_members[rng.integers(0, len(mule_members))]
            k = rng.integers(0, len(ring))
            src, dst = companies["company_id"].iloc[ring[k]], \
                companies["company_id"].iloc[ring[(k + 1) % len(ring)]]
            amount = float(rng.uniform(4.5e6, 5.5e6))  # just under 5m reporting-ish band
            tx_type = "transfer"
        elif kind < 0.10 and len(rt):
            src = dst = companies["company_id"].iloc[rng.choice(rt)]
            amount = float(rng.lognormal(16.5, 0.4))
            tx_type = "roundtrip"
        else:
            src, dst = rng.choice(companies["company_id"], 2, replace=False)
            amount = float(rng.lognormal(15.0, 1.2))
            tx_type = tx_types[rng.integers(0, len(tx_types))]
        rows.append((f"TX-{cfg.seed}-{i:08d}", i % (180 * 24 * 60), acct_of(src),
                     acct_of(dst), src, dst, round(amount, 2), tx_type))

    transactions = pd.DataFrame(rows, columns=[
        "tx_id", "minute_offset", "from_account", "to_account",
        "from_company", "to_company", "amount_ngn", "tx_type"])
    transactions["data_source"] = "SYNTHETIC"
    companies["data_source"] = "SYNTHETIC"
    accounts["data_source"] = "SYNTHETIC"
    return {"companies": companies, "accounts": accounts, "transactions": transactions}
