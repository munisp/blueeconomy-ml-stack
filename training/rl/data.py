"""Replay dataset builders for offline RL — REAL logged data only.

Each builder reads platform tables via a config-gated DSN and returns a
ReplayDataset of (context, action, reward) rows. There is NO synthetic
fallback: with no DSN configured the caller exits honestly
(training/rl/train.py), and with too few usable rows the builder raises
InsufficientHistory. Synthetic frames are constructed by tests/ only and are
passed straight to the build_* functions, never through a DSN default.

Replay schemas (read-only SQL against the real services' tables; every
table/column below was verified against the services' migrations):

queue-policy  (BEML_RL_QUEUE_PG_DSN -> port-interoperability DB):
    customs_declarations (db/migrations/0012_declarations.sql):
        declaration_id, submitted_at, cleared_at, is_aeo, risk_score,
        invoice_amount_minor, risk_lane
    action = the LOGGED lane assigned by the rules engine, mapped
        risk_lane -> {GREEN: 0, YELLOW: 1, RED: 2}
    reward — OUTCOME-SOURCE-GATED. There is NO inspection-outcome (`hit`)
        column anywhere in the platform schemas (verified: port-interop
        0001..0024, geo 0001..0019, singlewindow drizzle schema). The
        builder therefore REFUSES to build a reward unless an outcome
        source is explicitly configured (BEML_RL_QUEUE_OUTCOME_SOURCE, see
        QUEUE_OUTCOME_SOURCES). Until queue_policy_decisions / a real
        inspection-outcome feed accumulates, the only honest signal is
        clearance latency:
          "clearance-latency": reward = -0.05 * latency_hours
              (cleared_at - submitted_at, clipped to latency_cap_hours).
        With no outcome source configured the builder raises
        InsufficientHistory and the registry stays SCORING_UNAVAILABLE.

berth-allocation (BEML_RL_BERTH_PG_DSN -> geo-service DB):
    recommendation_log (geo db/migrations/0018_recommendation_log.sql).
    Verified: port_calls (port-interop 0001) carries ONLY lifecycle
    bookkeeping (call_id, vessel_imo, port_code, status, timestamps of
    record creation) — no arrived_at/berthed_at/departed_at/berth_id —
    and NO `berths` registry table exists in any migration of either
    service. The only berth-allocation DECISION log that exists is the
    Phase-18 shadow recommendation log. Per the migration's JOIN
    CONTRACT, `suggestion` holds the policy action and `outcome` the
    realized reward payload (filled later by the reward pipeline, keyed
    on recommendation_id). Until outcomes accumulate every build raises
    InsufficientHistory.
    action = suggestion->>'berthId' mapped to a stable index
    reward = -(waitingHours + turnaroundHours) / 24 from outcome JSONB
    contexts from request JSONB (port, n_vessels, n_berths) + cyclical
    time from created_at.

route-advice (BEML_RL_ROUTE_PG_DSN -> geo-service DB):
    recommendation_log WHERE kind='route_advice' AND outcome IS NOT NULL.
    action = suggestion->>'routeOption' (int index)
    reward = -(outcome->>'realizedDelayMin') / 60
    contexts = suggestion->>'predictedDelayMin', corridor (origin +
    '>' + destination from request JSONB), cyclical time from created_at.
    suggestion/outcome JSONB keys are the JOIN CONTRACT documented in
    0018: the reward pipeline writes realizedDelayMin per
    recommendation_id; rows with missing keys are dropped (never
    imputed), and below min_samples the builder raises
    InsufficientHistory.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class InsufficientHistory(RuntimeError):
    """Raised when the real logged data cannot support honest training."""


@dataclass
class ReplayDataset:
    """One batch of logged (context, action, reward) experience."""

    name: str
    contexts: np.ndarray          # (n, d) float32
    actions: np.ndarray           # (n,) int64 action indices
    rewards: np.ndarray           # (n,) float32
    feature_names: list[str]
    n_actions: int
    data_source: str              # provenance string, e.g. REAL:customs_declarations

    def __len__(self) -> int:
        return len(self.rewards)


def _connect(dsn: str):
    try:
        import psycopg
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise SystemExit(
            "RL training requires psycopg (pip install 'psycopg[binary]'); "
            "inference does not need it"
        ) from exc
    return psycopg.connect(dsn)


def load_frame(dsn: str, query: str) -> pd.DataFrame:
    with _connect(dsn) as conn:
        cur = conn.execute(query)
        cols = [d.name for d in cur.description]
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=cols)


# customs_declarations (port-interop 0012) — the REAL declaration queue
# history. Only terminal CLEARED rows with a logged lane and both lifecycle
# timestamps are replayable; status='CLEARED' guarantees cleared_at (DB CHECK).
QUEUE_QUERY = (
    "SELECT declaration_id, submitted_at, cleared_at, is_aeo, risk_score, "
    "invoice_amount_minor, risk_lane "
    "FROM customs_declarations "
    "WHERE status = 'CLEARED' AND submitted_at IS NOT NULL "
    "AND cleared_at IS NOT NULL AND risk_lane IS NOT NULL "
    "ORDER BY submitted_at"
)

# recommendation_log (geo 0018) is the ONLY berth-allocation decision log:
# port_calls has no berth lifecycle columns and no berths registry exists
# (verified against port-interop 0001..0024 and geo 0001..0019). JSONB keys
# follow the geo public request shape (camelCase, internal/api/recommend.go)
# and the 0018 JOIN CONTRACT (outcome filled later by the reward pipeline).
BERTH_QUERY = (
    "SELECT recommendation_id, created_at, "
    "request->>'portCode' AS port_code, "
    "jsonb_array_length(request->'vessels') AS n_vessels, "
    "jsonb_array_length(COALESCE(request->'berths', '[]'::jsonb)) AS n_berths, "
    "suggestion->>'berthId' AS berth_id, "
    "(outcome->>'waitingHours')::float8 AS waiting_hours, "
    "(outcome->>'turnaroundHours')::float8 AS turnaround_hours "
    "FROM recommendation_log "
    "WHERE kind = 'berth_allocation' AND outcome IS NOT NULL "
    "ORDER BY created_at"
)

# recommendation_log (geo 0018): route advice with realized outcomes.
ROUTE_QUERY = (
    "SELECT recommendation_id, created_at, "
    "request->>'origin' AS origin, "
    "request->>'destination' AS destination, "
    "(suggestion->>'routeOption')::int AS route_option, "
    "(suggestion->>'predictedDelayMin')::float8 AS predicted_delay_min, "
    "(outcome->>'realizedDelayMin')::float8 AS realized_delay_min, "
    "EXTRACT(hour FROM created_at)::int AS hour, "
    "(EXTRACT(isodow FROM created_at)::int - 1) AS dow "
    "FROM recommendation_log "
    "WHERE kind = 'route_advice' AND outcome IS NOT NULL "
    "ORDER BY created_at"
)

# Logged-lane -> action index. GREEN is the rules engine's fast lane, RED is
# manual review; the mapping is contractual (serving returns the index).
RISK_LANE_ACTIONS = {"GREEN": 0, "YELLOW": 1, "RED": 2}

# Honest queue-outcome sources. There is NO inspection-hit signal in any
# platform schema; do not add one here until a real column/table exists.
QUEUE_OUTCOME_CLEARANCE_LATENCY = "clearance-latency"
QUEUE_OUTCOME_SOURCES = (QUEUE_OUTCOME_CLEARANCE_LATENCY,)


def _cyclical(hour: np.ndarray, dow: np.ndarray) -> np.ndarray:
    return np.column_stack([
        np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
    ])


def _require(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise InsufficientHistory(
            f"{name}: logged table is missing required columns {missing}; "
            f"refusing to train on a partial schema (fail-closed)."
        )


def build_queue_replay(df: pd.DataFrame, min_samples: int = 500,
                       latency_cap_hours: float = 72.0,
                       outcome_source: str | None = None) -> ReplayDataset:
    """(context, action, reward) from customs_declarations (port-interop 0012).

    action = logged risk_lane mapped GREEN->0, YELLOW->1, RED->2 (the lane
        the rules engine actually assigned).
    reward = outcome-source-gated. No inspection-outcome (`hit`) column
        exists in any platform schema, so a reward may ONLY be built from an
        explicitly configured outcome source:
          "clearance-latency": reward = -0.05 * latency_hours (capped).
        outcome_source=None (or unknown) raises InsufficientHistory — the
        task stays SCORING_UNAVAILABLE rather than fabricating a reward.
    invoice_amount_minor is minor units; converted to major (/100) before
        log1p so the feature matches the historical declared_value parity.
    """
    name = "queue-policy"
    if outcome_source not in QUEUE_OUTCOME_SOURCES:
        raise InsufficientHistory(
            f"INSUFFICIENT_HISTORY: {name} has no configured outcome source "
            f"(BEML_RL_QUEUE_OUTCOME_SOURCE is unset or not one of "
            f"{QUEUE_OUTCOME_SOURCES}). No inspection-outcome column exists "
            f"in the declaration schema, so a reward cannot be derived "
            f"honestly until queue_policy_decisions / a real outcome feed "
            f"accumulates. Registry stays SCORING_UNAVAILABLE.")
    _require(df, ["submitted_at", "cleared_at", "is_aeo", "risk_score",
                  "invoice_amount_minor", "risk_lane"], name)
    df = df.dropna(subset=["cleared_at", "risk_lane", "submitted_at"]).copy()
    lanes = df["risk_lane"].astype(str).str.upper()
    unknown = sorted(set(lanes.unique()) - set(RISK_LANE_ACTIONS))
    if unknown:
        raise InsufficientHistory(
            f"{name}: unmapped risk_lane values {unknown}; refusing to "
            f"guess an action index (fail-closed).")
    actions = lanes.map(RISK_LANE_ACTIONS).to_numpy(np.int64)
    submitted = pd.to_datetime(df["submitted_at"], utc=True)
    cleared = pd.to_datetime(df["cleared_at"], utc=True)
    latency_h = ((cleared - submitted).dt.total_seconds() / 3600.0
                 ).clip(0, latency_cap_hours).to_numpy(np.float64)
    # clearance-latency outcome: faster clearance is better; nothing else is
    # claimed (no violation-hit term exists in the schema).
    reward = (-0.05 * latency_h).astype(np.float32)
    hour = submitted.dt.hour + submitted.dt.minute / 60.0
    cyc = _cyclical(hour.to_numpy(np.float64),
                    submitted.dt.dayofweek.to_numpy(np.float64))
    # NOTE: realized latency is the reward input, never a decision context.
    invoice_major = df["invoice_amount_minor"].to_numpy(np.float64) / 100.0
    contexts = np.column_stack([
        df["risk_score"].to_numpy(np.float64),
        df["is_aeo"].astype(float).to_numpy(np.float64),
        np.log1p(np.clip(invoice_major, 0, None)),
        cyc,
        np.ones(len(df)),  # bias term (linear policies need an intercept)
    ]).astype(np.float32)
    features = ["risk_score", "is_aeo", "log_invoice_value_major",
                "hour_sin", "hour_cos", "dow_sin", "dow_cos", "bias"]
    ds = ReplayDataset(name=name, contexts=contexts, actions=actions,
                       rewards=reward,
                       feature_names=features, n_actions=3,
                       data_source="REAL:customs_declarations")
    if len(ds) < min_samples:
        raise InsufficientHistory(
            f"INSUFFICIENT_HISTORY: {len(ds)} usable queue rows "
            f"(< {min_samples}); no policy trained, registry stays "
            f"SCORING_UNAVAILABLE.")
    return ds


def build_berth_replay(df: pd.DataFrame, min_samples: int = 500) -> ReplayDataset:
    """(context, action, reward) from recommendation_log (geo 0018).

    Verified schema reality: port_calls (port-interop 0001) has no berth
    lifecycle columns (arrived_at/berthed_at/departed_at/berth_id) and no
    `berths` registry table exists in any migration — the Phase-18 shadow
    recommendation log is the only berth-allocation decision trail.

    action = suggestion->>'berthId' (stable index over observed berths)
    reward = -(waitingHours + turnaroundHours) / 24 from outcome JSONB,
        i.e. total port time in days, negated.
    Outcomes are populated later by the reward pipeline (0018 JOIN
    CONTRACT); until enough realized outcomes exist this raises
    InsufficientHistory and the registry stays SCORING_UNAVAILABLE.
    """
    name = "berth-allocation"
    _require(df, ["created_at", "port_code", "n_vessels", "n_berths",
                  "berth_id", "waiting_hours", "turnaround_hours"], name)
    df = df.dropna(subset=["berth_id", "waiting_hours", "turnaround_hours",
                           "created_at"]).copy()
    created = pd.to_datetime(df["created_at"], utc=True)
    wait_h = df["waiting_hours"].to_numpy(np.float64).clip(0)
    turn_h = df["turnaround_hours"].to_numpy(np.float64).clip(0)
    reward = (-(wait_h + turn_h) / 24.0).astype(np.float32)
    berth_ids = sorted(df["berth_id"].astype(str).unique())
    berth_index = {b: i for i, b in enumerate(berth_ids)}
    if len(berth_ids) < 2:
        raise InsufficientHistory(
            f"INSUFFICIENT_HISTORY: only {len(berth_ids)} distinct berth(s) "
            f"observed; there is no allocation decision to learn. Registry "
            f"stays SCORING_UNAVAILABLE.")
    actions = df["berth_id"].astype(str).map(berth_index).to_numpy(np.int64)
    hour = created.dt.hour + created.dt.minute / 60.0
    cyc = _cyclical(hour.to_numpy(np.float64),
                    created.dt.dayofweek.to_numpy(np.float64))
    contexts = np.column_stack([
        df["n_vessels"].to_numpy(np.float64),
        df["n_berths"].to_numpy(np.float64),
        cyc,
        np.ones(len(df)),  # bias term
    ]).astype(np.float32)
    features = ["n_vessels", "n_berths",
                "hour_sin", "hour_cos", "dow_sin", "dow_cos", "bias"]
    ds = ReplayDataset(name=name, contexts=contexts, actions=actions,
                       rewards=reward, feature_names=features,
                       n_actions=len(berth_ids),
                       data_source="REAL:recommendation_log")
    if len(ds) < min_samples:
        raise InsufficientHistory(
            f"INSUFFICIENT_HISTORY: {len(ds)} realized berth-allocation "
            f"outcomes (< {min_samples}); no policy trained, registry "
            f"stays SCORING_UNAVAILABLE.")
    return ds


def build_route_replay(df: pd.DataFrame, min_samples: int = 500) -> ReplayDataset:
    """(context, action, reward) from recommendation_log (geo 0018).

    action = suggestion->>'routeOption' index; reward =
        -(outcome->>'realizedDelayMin') / 60. corridor is derived from the
        request JSONB (origin + '>' + destination).
    Requires the reward pipeline to have accumulated realized outcomes
    (0018 JOIN CONTRACT); otherwise InsufficientHistory and the model
    stays SCORING_UNAVAILABLE.
    """
    name = "route-advice"
    _require(df, ["route_option", "predicted_delay_min", "realized_delay_min",
                  "hour", "dow", "origin", "destination"], name)
    df = df.dropna(subset=["realized_delay_min", "route_option"]).copy()
    df["corridor"] = (df["origin"].astype(str) + ">"
                      + df["destination"].astype(str))
    corridors = sorted(df["corridor"].unique())
    corridor_index = {c: i / max(1, len(corridors)) for i, c in enumerate(corridors)}
    options = sorted(df["route_option"].unique())
    if len(options) < 2:
        raise InsufficientHistory(
            "INSUFFICIENT_HISTORY: a single route option observed; nothing "
            "to learn. Registry stays SCORING_UNAVAILABLE.")
    option_index = {o: i for i, o in enumerate(options)}
    actions = df["route_option"].map(option_index).to_numpy(np.int64)
    reward = (-df["realized_delay_min"].to_numpy(np.float64) / 60.0).astype(np.float32)
    cyc = _cyclical(df["hour"].to_numpy(np.float64), df["dow"].to_numpy(np.float64))
    contexts = np.column_stack([
        df["predicted_delay_min"].to_numpy(np.float64),
        df["corridor"].map(corridor_index).to_numpy(np.float64),
        cyc,
        np.ones(len(df)),  # bias term
    ]).astype(np.float32)
    features = ["predicted_delay_min", "corridor_index_norm",
                "hour_sin", "hour_cos", "dow_sin", "dow_cos", "bias"]
    ds = ReplayDataset(name=name, contexts=contexts, actions=actions,
                       rewards=reward, feature_names=features,
                       n_actions=len(options),
                       data_source="REAL:recommendation_log")
    if len(ds) < min_samples:
        raise InsufficientHistory(
            f"INSUFFICIENT_HISTORY: {len(ds)} realized route outcomes "
            f"(< {min_samples}); no policy trained, registry stays "
            f"SCORING_UNAVAILABLE.")
    return ds
