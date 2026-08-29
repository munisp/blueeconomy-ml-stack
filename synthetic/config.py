"""Config-driven volumes and seeds for the synthetic generators.

All knobs are dataclasses so they can be constructed from YAML/CLI without
any framework dependency. Every config carries an explicit seed: generation
is byte-for-byte reproducible given the same seed + library versions.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json


@dataclass(frozen=True)
class DeclarationConfig:
    n_declarations: int = 20_000
    n_consignees: int = 2_500
    n_shell_clusters: int = 60
    shell_cluster_size: int = 8
    fraud_rate: float = 0.07          # overall fraction of fraudulent declarations
    undervalue_share: float = 0.5     # of fraud: undervaluation
    misclassify_share: float = 0.3    # of fraud: HS misclassification
    # remainder is shell-consignee routing
    seed: int = 20240


@dataclass(frozen=True)
class AisConfig:
    n_vessels: int = 120
    days: int = 21
    pings_per_hour: float = 2.0       # mean underway ping rate
    dark_gap_rate: float = 0.10       # fraction of vessels with spoofing gaps
    loiter_rate: float = 0.18         # fraction with suspicious loitering
    eez_incursion_rate: float = 0.06  # fraction with EEZ incursions
    seed: int = 30311


@dataclass(frozen=True)
class CvffConfig:
    n_companies: int = 1_800
    n_accounts: int = 2_600
    n_transactions: int = 40_000
    n_mule_rings: int = 24
    mule_ring_size: int = 6
    roundtrip_rate: float = 0.012     # fraction of companies in round-tripping
    seed: int = 40412


@dataclass(frozen=True)
class SyntheticConfig:
    declarations: DeclarationConfig = field(default_factory=DeclarationConfig)
    ais: AisConfig = field(default_factory=AisConfig)
    cvff: CvffConfig = field(default_factory=CvffConfig)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)
