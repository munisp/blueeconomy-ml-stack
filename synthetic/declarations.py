"""SYNTHETIC Nigerian customs declaration generator.

Produces a statistically realistic mirror of the platform declaration schema
used by the customs/manifest domain. Fraud patterns embedded with known ground
truth labels (this is why the data is useful for training):

- undervaluation: declared unit value is a fraction of the HS reference price
- HS misclassification: goods are declared under a lower-duty HS sibling code
- shell-consignee clusters: groups of TINs sharing officers/addresses that
  route fraud disproportionately

DATA IS SYNTHETIC. Every row carries data_source='SYNTHETIC'.
"""

from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd

from .config import DeclarationConfig

# Nigerian ports (UN/LOCODE) served by NPA.
PORTS = ["NGAPP", "NGTIN", "NGONN", "NGCBQ", "NGLOS", "NGKOK"]
PORT_NAMES = {
    "NGAPP": "Lagos Apapa", "NGTIN": "Tin Can Island", "NGONN": "Onne",
    "NGCBQ": "Calabar", "NGLOS": "Lagos (Lekki)", "NGKOK": "Koko",
}

# A small but realistic HS catalogue: (hs6, description, duty_rate,
# reference_price_usd_per_kg, typical_weight_kg). Duty rates reflect the
# Nigerian CET bands (0/5/10/20/35%).
HS_CATALOGUE = [
    ("030243", "Sardines, frozen", 0.20, 1.1, 27_000),
    ("100630", "Semi-milled rice", 0.35, 0.55, 28_000),
    ("170199", "Refined sugar", 0.20, 0.60, 27_500),
    ("271012", "Motor spirit (PMS)", 0.20, 0.85, 30_000),
    ("300490", "Medicaments, retail", 0.10, 48.0, 4_000),
    ("390410", "PVC resins", 0.10, 1.3, 24_000),
    ("401120", "Truck/bus tyres, new", 0.20, 2.4, 18_000),
    ("520852", "Printed cotton fabrics", 0.35, 6.5, 12_000),
    ("851712", "Smartphones", 0.05, 320.0, 900),
    ("870333", "Used vehicles >2500cc", 0.35, 9.0, 2_100),
    ("940360", "Wooden furniture", 0.35, 3.2, 8_000),
    ("847130", "Laptop computers", 0.00, 410.0, 1_200),
]

ORIGINS = ["CN", "IN", "TR", "BR", "TH", "VN", "GB", "US", "ZA", "AE", "NL", "KR"]

FX_NGN_PER_USD = 1550.0  # indicative 2024-2025 official-ish rate, fixed for reproducibility


def _tin(rng: np.random.Generator) -> str:
    return f"{rng.integers(10**7, 10**8):08d}-{rng.integers(10, 99):04d}"


def _shell_tin(cluster_id: int, member: int) -> str:
    # Shell TINs are deterministic functions of the cluster so the GNN can
    # learn cluster structure; clearly synthetic format.
    digest = hashlib.sha256(f"shell-{cluster_id}-{member}".encode()).hexdigest()
    return f"{int(digest[:8], 16) % 10**8:08d}-{int(digest[8:10], 16) % 90 + 10:04d}"


def generate_declarations(cfg: DeclarationConfig = DeclarationConfig()) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_declarations

    # --- consignee population: legitimate TINs + shell clusters ---
    legit_tins = [_tin(rng) for _ in range(cfg.n_consignees)]
    shell_members: dict[str, int] = {}
    for c in range(cfg.n_shell_clusters):
        for m in range(cfg.shell_cluster_size):
            shell_members[_shell_tin(c, m)] = c
    all_tins = legit_tins + list(shell_members)

    hs_idx = rng.integers(0, len(HS_CATALOGUE), n)
    cat = [HS_CATALOGUE[i] for i in hs_idx]
    true_hs = np.array([c[0] for c in cat])
    duty_rates = np.array([c[1 + 1] for c in cat], dtype=float)
    ref_price = np.array([c[3] for c in cat], dtype=float)
    typ_weight = np.array([c[4] for c in cat], dtype=float)

    weights = np.maximum(50.0, rng.lognormal(np.log(typ_weight), 0.35))
    unit_value = ref_price * rng.lognormal(0.0, 0.22, n)  # honest price noise
    declared_hs = true_hs.copy()
    declared_duty = duty_rates.copy()

    # --- assign fraud ---
    is_fraud = rng.random(n) < cfg.fraud_rate
    fraud_type = np.array(["none"] * n, dtype=object)
    r = rng.random(n)
    undervalue_mask = is_fraud & (r < cfg.undervalue_share)
    misclassify_mask = is_fraud & ~undervalue_mask & (
        r < cfg.undervalue_share + cfg.misclassify_share)
    shell_mask = is_fraud & ~undervalue_mask & ~misclassify_mask

    # undervaluation: declare a fraction of true unit value
    unit_value[undervalue_mask] *= rng.uniform(0.25, 0.6, undervalue_mask.sum())
    fraud_type[undervalue_mask] = "undervaluation"

    # misclassification: swap to a lower-duty HS code (nearest lower-duty sibling)
    lower_duty = sorted({c[0] for c in HS_CATALOGUE if c[2] <= 0.10})
    for i in np.where(misclassify_mask)[0]:
        if duty_rates[i] > 0.10:
            declared_hs[i] = lower_duty[rng.integers(0, len(lower_duty))]
            declared_duty[i] = next(c[2] for c in HS_CATALOGUE if c[0] == declared_hs[i])
        else:
            # already lowest band -> misdeclared description instead
            unit_value[i] *= rng.uniform(0.5, 0.75)
    fraud_type[misclassify_mask] = "hs_misclassification"

    # shell routing: fraudulent declarations go through shell TINs
    fraud_type[shell_mask] = "shell_consignee"

    # --- consignee assignment ---
    shell_tins = list(shell_members)
    consignees = np.array(rng.choice(all_tins, n), dtype=object)
    shell_pick = rng.integers(0, len(shell_tins), shell_mask.sum())
    consignees[shell_mask] = [shell_tins[i] for i in shell_pick]
    # a few shell TINs also appear in legitimate traffic (camouflage)
    camo = (~is_fraud) & (rng.random(n) < 0.01)
    consignees[camo] = [shell_tins[i] for i in rng.integers(0, len(shell_tins), camo.sum())]

    cif_usd = unit_value * weights
    duty_paid_ngn = cif_usd * declared_duty * FX_NGN_PER_USD

    # behavioural/risk features (what the tabular model sees)
    days_ago = rng.integers(0, 365, n)
    declarant_experience = rng.integers(0, 2, n)  # 1 = established broker
    prior_declarations = rng.poisson(is_fraud * 2 + 8, n)
    night_filing = (rng.random(n) < (0.30 * is_fraud + 0.06)).astype(int)

    df = pd.DataFrame({
        "declaration_id": [f"SYND-{cfg.seed}-{i:07d}" for i in range(n)],
        "days_ago": days_ago,
        "port_code": rng.choice(PORTS, n, p=[0.34, 0.26, 0.16, 0.08, 0.11, 0.05]),
        "hs_code_declared": declared_hs,
        "hs_code_true": true_hs,
        "country_of_origin": rng.choice(ORIGINS, n,
                                        p=[0.30, 0.14, 0.09, 0.08, 0.07, 0.06,
                                           0.05, 0.05, 0.05, 0.05, 0.03, 0.03]),
        "consignee_tin": consignees,
        "consignee_is_shell": np.isin(consignees, shell_tins).astype(int),
        "declarant_is_established": declarant_experience,
        "declarant_prior_declarations": prior_declarations,
        "night_filing": night_filing,
        "weight_kg": np.round(weights, 1),
        "unit_value_usd_per_kg": np.round(unit_value, 4),
        "reference_price_usd_per_kg": ref_price,
        "price_ratio_vs_reference": np.round(unit_value / ref_price, 4),
        "cif_value_usd": np.round(cif_usd, 2),
        "duty_rate_applied": declared_duty,
        "duty_rate_true": duty_rates,
        "duty_paid_ngn": np.round(duty_paid_ngn, 2),
        "fx_rate_ngn_usd": FX_NGN_PER_USD,
        "is_fraud": is_fraud.astype(int),
        "fraud_type": fraud_type,
        "data_source": "SYNTHETIC",
    })
    return df


FEATURE_COLUMNS = [
    "price_ratio_vs_reference", "log_cif_usd", "duty_rate_applied",
    "log_weight_kg", "consignee_is_shell", "declarant_is_established",
    "declarant_prior_declarations", "night_filing", "hs_mismatch",
    "port_enc", "origin_enc",
]


def to_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Feature matrix + label vector for the tabular trainer."""
    x = pd.DataFrame({
        "price_ratio_vs_reference": df["price_ratio_vs_reference"],
        "log_cif_usd": np.log1p(df["cif_value_usd"]),
        "duty_rate_applied": df["duty_rate_applied"],
        "log_weight_kg": np.log1p(df["weight_kg"]),
        "consignee_is_shell": df["consignee_is_shell"],
        "declarant_is_established": df["declarant_is_established"],
        "declarant_prior_declarations": df["declarant_prior_declarations"],
        "night_filing": df["night_filing"],
        "hs_mismatch": (df["hs_code_declared"] != df["hs_code_true"]).astype(int),
        "port_enc": df["port_code"].map({p: i for i, p in enumerate(PORTS)}).astype(float),
        "origin_enc": df["country_of_origin"].map({o: i for i, o in enumerate(ORIGINS)}).astype(float),
    })
    return x[FEATURE_COLUMNS].to_numpy(dtype=np.float32), df["is_fraud"].to_numpy(dtype=np.int64)
