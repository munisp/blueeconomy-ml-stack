"""Replay dataset builders for offline RL — REAL logged data only.

Each builder reads platform tables via a config-gated DSN and returns a
ReplayDataset of (context, action, reward) rows. There is NO synthetic
fallback: with no DSN configured the caller exits honestly
(training/rl/train.py), and with too few usable rows the builder raises
InsufficientHistory. Synthetic frames are constructed by tests/ only and are
passed straight to the build_* functions, never through a DSN default.

Replay schemas (read-only SQL against the real services' tables):

queue-policy  (BEML_RL_QUEUE_PG_DSN, singlewindow declaration queue history):
    declaration_id, submitted_at, processed_at, aeo_status, risk_score,
    declared_value, priority_bucket  (the LOGGED action taken by the
    officer/rules engine: 0=standard, 1=fast-lane, 2=manual-review),
    hit  (1 if inspection found a real violation, else 0)
  reward = hit - latency_penalty(processed_at - submitted_at)

berth-allocation (BEML_RL_BERTH_PG_DSN, geo-service + port-interop
berth+calls tables):
    call_id, arrived_at, berthed_at, departed_at, berth_id, vessel_loa_m,
    vessel_gt, queue_len_at_arrival, berth_depth_m, berth_length_m,
    n_berths
  action = berth_id mapped to a stable index; reward = -turnaround_hours
  (berthed_at..departed_at) minus waiting_hours (arrived_at..berthed_at),
  normalised; contexts include queue state and vessel/berth compatibility.

route-advice (BEML_RL_ROUTE_PG_DSN, geo-service route-advice logs):
    advice_id, route_option, predicted_delay_min, realized_delay_min,
    hour, dow, corridor  (one row per issued recommendation whose realized
    outcome was later logged)
  action = route_option index; reward = -(realized_delay_min).
  This table is populated by W2's recommendation logging; until enough
  realized outcomes exist the builder raises InsufficientHistory and the
  model stays SCORING_UNAVAILABLE.
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
    data_source: str              # provenance string, e.g. REAL:declarations_queue

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


QUEUE_QUERY = (
    "SELECT declaration_id, submitted_at, processed_at, aeo_status, "
    "risk_score, declared_value, priority_bucket, hit "
    "FROM declaration_queue_history WHERE processed_at IS NOT NULL "
    "ORDER BY submitted_at"
)

BERTH_QUERY = (
    "SELECT c.call_id, c.arrived_at, c.berthed_at, c.departed_at, "
    "c.berth_id, c.vessel_loa_m, c.vessel_gt, c.queue_len_at_arrival, "
    "b.depth_m AS berth_depth_m, b.length_m AS berth_length_m "
    "FROM port_calls c JOIN berths b ON b.berth_id = c.berth_id "
    "WHERE c.berthed_at IS NOT NULL AND c.departed_at IS NOT NULL "
    "ORDER BY c.arrived_at"
)

ROUTE_QUERY = (
    "SELECT advice_id, route_option, predicted_delay_min, realized_delay_min, "
    "hour, dow, corridor FROM route_advice_outcomes "
    "WHERE realized_delay_min IS NOT NULL ORDER BY advice_id"
)


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
                       latency_cap_hours: float = 72.0) -> ReplayDataset:
    """(context, action, reward) from declaration queue history.

    action = logged priority_bucket (0 standard, 1 fast-lane, 2 manual-review)
    reward = hit - 0.05 * latency_hours (capped); a fast-lane decision that
    catches a violation is rewarded, slow clearances are penalised.
    """
    name = "queue-policy"
    _require(df, ["submitted_at", "processed_at", "aeo_status", "risk_score",
                  "declared_value", "priority_bucket", "hit"], name)
    df = df.dropna(subset=["processed_at", "priority_bucket", "hit"]).copy()
    submitted = pd.to_datetime(df["submitted_at"], utc=True)
    processed = pd.to_datetime(df["processed_at"], utc=True)
    latency_h = ((processed - submitted).dt.total_seconds() / 3600.0
                 ).clip(0, latency_cap_hours).to_numpy(np.float64)
    reward = df["hit"].to_numpy(np.float64) - 0.05 * latency_h
    hour = submitted.dt.hour + submitted.dt.minute / 60.0
    cyc = _cyclical(hour.to_numpy(np.float64),
                    submitted.dt.dayofweek.to_numpy(np.float64))
    # NOTE: realized latency is the reward input, never a decision context.
    contexts = np.column_stack([
        df["risk_score"].to_numpy(np.float64),
        df["aeo_status"].astype(float).to_numpy(np.float64),
        np.log1p(df["declared_value"].to_numpy(np.float64)),
        cyc,
        np.ones(len(df)),  # bias term (linear policies need an intercept)
    ]).astype(np.float32)
    features = ["risk_score", "aeo_status", "log_declared_value",
                "hour_sin", "hour_cos", "dow_sin", "dow_cos", "bias"]
    actions = df["priority_bucket"].to_numpy(np.int64)
    ds = ReplayDataset(name=name, contexts=contexts, actions=actions,
                       rewards=reward.astype(np.float32),
                       feature_names=features, n_actions=3,
                       data_source="REAL:declaration_queue_history")
    if len(ds) < min_samples:
        raise InsufficientHistory(
            f"INSUFFICIENT_HISTORY: {len(ds)} usable queue rows "
            f"(< {min_samples}); no policy trained, registry stays "
            f"SCORING_UNAVAILABLE.")
    return ds


def build_berth_replay(df: pd.DataFrame, min_samples: int = 500) -> ReplayDataset:
    """(context, action, reward) from port-call / berth logs.

    action = chosen berth (stable index over observed berth_ids)
    reward = -(waiting_hours + turnaround_hours) / 24, i.e. total port time
    in days, negated. Contexts carry queue state and vessel/berth geometry so
    the policy can learn feasibility-aware allocation.
    """
    name = "berth-allocation"
    _require(df, ["arrived_at", "berthed_at", "departed_at", "berth_id",
                  "vessel_loa_m", "vessel_gt", "queue_len_at_arrival",
                  "berth_depth_m", "berth_length_m"], name)
    df = df.dropna(subset=["berthed_at", "departed_at", "berth_id"]).copy()
    arrived = pd.to_datetime(df["arrived_at"], utc=True)
    berthed = pd.to_datetime(df["berthed_at"], utc=True)
    departed = pd.to_datetime(df["departed_at"], utc=True)
    wait_h = ((berthed - arrived).dt.total_seconds() / 3600.0).clip(0)
    turn_h = ((departed - berthed).dt.total_seconds() / 3600.0).clip(0)
    reward = -((wait_h + turn_h) / 24.0).to_numpy(np.float32)
    berth_ids = sorted(df["berth_id"].unique())
    berth_index = {b: i for i, b in enumerate(berth_ids)}
    if len(berth_ids) < 2:
        raise InsufficientHistory(
            f"INSUFFICIENT_HISTORY: only {len(berth_ids)} distinct berth(s) "
            f"observed; there is no allocation decision to learn. Registry "
            f"stays SCORING_UNAVAILABLE.")
    actions = df["berth_id"].map(berth_index).to_numpy(np.int64)
    hour = arrived.dt.hour + arrived.dt.minute / 60.0
    cyc = _cyclical(hour.to_numpy(np.float64),
                    arrived.dt.dayofweek.to_numpy(np.float64))
    contexts = np.column_stack([
        df["queue_len_at_arrival"].to_numpy(np.float64),
        np.log1p(df["vessel_loa_m"].to_numpy(np.float64)),
        np.log1p(df["vessel_gt"].to_numpy(np.float64)),
        df["berth_depth_m"].to_numpy(np.float64),
        df["berth_length_m"].to_numpy(np.float64),
        cyc,
        np.ones(len(df)),  # bias term
    ]).astype(np.float32)
    features = ["queue_len_at_arrival", "log_vessel_loa_m", "log_vessel_gt",
                "berth_depth_m", "berth_length_m",
                "hour_sin", "hour_cos", "dow_sin", "dow_cos", "bias"]
    ds = ReplayDataset(name=name, contexts=contexts, actions=actions,
                       rewards=reward, feature_names=features,
                       n_actions=len(berth_ids),
                       data_source="REAL:port_calls+berths")
    if len(ds) < min_samples:
        raise InsufficientHistory(
            f"INSUFFICIENT_HISTORY: {len(ds)} usable port calls "
            f"(< {min_samples}); no policy trained, registry stays "
            f"SCORING_UNAVAILABLE.")
    return ds


def build_route_replay(df: pd.DataFrame, min_samples: int = 500) -> ReplayDataset:
    """(context, action, reward) from logged route advice + realized outcomes.

    action = advised route_option index; reward = -realized_delay_min / 60.
    Requires W2's recommendation logging to have accumulated realized
    outcomes; otherwise InsufficientHistory and the model stays unavailable.
    """
    name = "route-advice"
    _require(df, ["route_option", "predicted_delay_min", "realized_delay_min",
                  "hour", "dow", "corridor"], name)
    df = df.dropna(subset=["realized_delay_min", "route_option"]).copy()
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
                       data_source="REAL:route_advice_outcomes")
    if len(ds) < min_samples:
        raise InsufficientHistory(
            f"INSUFFICIENT_HISTORY: {len(ds)} realized route outcomes "
            f"(< {min_samples}); no policy trained, registry stays "
            f"SCORING_UNAVAILABLE.")
    return ds
