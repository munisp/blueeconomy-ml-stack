"""Offline RL training entrypoint — config-gated, OPE-gated, fail-closed.

Per task:
- queue-policy      LinUCB contextual bandit  (BEML_RL_QUEUE_PG_DSN)
- berth-allocation  conservative CQL-H        (BEML_RL_BERTH_PG_DSN)
- route-advice      LinUCB contextual bandit  (BEML_RL_ROUTE_PG_DSN)

Hard rules (no exceptions):
- DSN unset                    -> honest exit, training disabled; the serving
                                  registry keeps reporting SCORING_UNAVAILABLE.
- InsufficientHistory          -> honest exit, nothing exported.
- OPE gate fails (DR estimate does not beat the logged baseline policy by
  --min-delta, or propensity support is too thin) -> honest exit, NOTHING is
  exported; a candidate policy that cannot demonstrably beat what the
  platform already does is never promoted.
- No synthetic data anywhere in this path; tests/ build their own toy frames.

Algorithm rationale (also in the module docstrings):
- queue-policy / route-advice are single-step decisions with no decision-
  caused state transition -> contextual bandit (LinUCB), not sequential RL.
- berth-allocation couples decisions through berth occupancy -> sequential
  problem -> offline-constrained CQL-H; gamma=0 by default because logs do
  not provide a verified next-state join (stated in metrics.json).

Usage:
    BEML_RL_QUEUE_PG_DSN=postgres://... \
    python -m training.rl.train --task queue-policy \
        --out models/queue-policy --version 0.1.0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from training.common import RunTracker  # noqa: E402
from training.rl import data as rl_data  # noqa: E402
from training.rl import ope as rl_ope  # noqa: E402
from training.rl.bandits import LinUCB  # noqa: E402
from training.rl.cql import (export_linear_policy_onnx, export_policy_onnx,  # noqa: E402
                             greedy_policy_actions, train_cql)

TASKS = {
    "queue-policy": {
        "dsn_env": "BEML_RL_QUEUE_PG_DSN",
        "query": rl_data.QUEUE_QUERY,
        "builder": rl_data.build_queue_replay,
        "algorithm": "linucb-contextual-bandit",
    },
    "berth-allocation": {
        "dsn_env": "BEML_RL_BERTH_PG_DSN",
        "query": rl_data.BERTH_QUERY,
        "builder": rl_data.build_berth_replay,
        "algorithm": "cql-h-conservative-q-learning",
    },
    "route-advice": {
        "dsn_env": "BEML_RL_ROUTE_PG_DSN",
        "query": rl_data.ROUTE_QUERY,
        "builder": rl_data.build_route_replay,
        "algorithm": "linucb-contextual-bandit",
    },
}


class PromotionRefused(RuntimeError):
    """The OPE gate refused promotion; nothing may be exported."""


def load_replay(task: str, args: argparse.Namespace) -> rl_data.ReplayDataset:
    spec = TASKS[task]
    dsn = os.environ.get(spec["dsn_env"], "").strip()
    if not dsn:
        raise SystemExit(
            f"{spec['dsn_env']} is not set: {task} RL training is "
            f"config-gated and disabled. The serving registry continues to "
            f"report SCORING_UNAVAILABLE for {task} (fail-closed). "
            f"There is no synthetic training default.")
    frame = rl_data.load_frame(dsn, spec["query"])
    return spec["builder"](frame, min_samples=args.min_samples)


def fit_policy(replay: rl_data.ReplayDataset, algorithm: str,
               args: argparse.Namespace):
    """Trains on the FIRST --train-frac of rows (time-ordered logs)."""
    n = len(replay)
    split = int(n * args.train_frac)
    x_tr, a_tr, r_tr = (replay.contexts[:split], replay.actions[:split],
                        replay.rewards[:split])
    extra = {}
    if algorithm == "linucb-contextual-bandit":
        bandit = LinUCB(replay.n_actions, replay.contexts.shape[1],
                        alpha=args.alpha_ucb).fit(x_tr, a_tr, r_tr)

        def policy_fn(ctx):
            return bandit.greedy_actions(ctx)

        def export_fn(path):
            export_linear_policy_onnx(bandit.theta(), path)
    else:
        qnet, extra = train_cql(
            x_tr, a_tr, r_tr, replay.n_actions,
            gamma=args.gamma, alpha=args.alpha_cql, epochs=args.epochs,
            seed=args.seed)

        def policy_fn(ctx):
            return greedy_policy_actions(qnet, ctx)

        def export_fn(path):
            export_policy_onnx(qnet, replay.contexts.shape[1], path)
    return policy_fn, export_fn, extra


def train(args: argparse.Namespace) -> dict:
    replay = load_replay(args.task, args)
    algorithm = TASKS[args.task]["algorithm"]
    n = len(replay)
    split = int(n * args.train_frac)
    if n - split < 50:
        raise SystemExit(
            f"INSUFFICIENT_HISTORY: only {n - split} held-out rows for OPE "
            f"(need >= 50); refusing to gate a policy on noise. Registry "
            f"stays SCORING_UNAVAILABLE.")
    policy_fn, export_fn, extra = fit_policy(replay, algorithm, args)
    # OPE gate on the held-out time-ordered tail (never trained on).
    x_te, a_te, r_te = (replay.contexts[split:], replay.actions[split:],
                        replay.rewards[split:])
    cand_actions = policy_fn(x_te)
    report = rl_ope.evaluate(cand_actions, x_te, a_te, r_te, replay.n_actions,
                             min_delta=args.min_delta)
    with RunTracker(args.task, f"{args.task}-{args.version}") as track:
        track.log_params({
            "model": args.task, "version": args.version,
            "algorithm": algorithm, "seed": args.seed,
            "n_train": split, "n_ope": n - split,
            "n_actions": replay.n_actions,
            "features": ",".join(replay.feature_names),
            "data_source": replay.data_source,
        })
        track.log_metrics({
            "ope.dr_candidate": report.dr_candidate,
            "ope.ips_candidate": report.ips_candidate,
            "ope.baseline_mean_reward": report.baseline_mean_reward,
            "ope.match_rate": report.match_rate,
            "ope.promoted": float(report.promoted),
        })
        if not report.promoted:
            # The gate refused. Honest exit: no artifact, registry untouched.
            raise SystemExit(
                f"OPE_GATE_REFUSED: {report.reason}. Nothing exported; "
                f"/score/{args.task} keeps reporting SCORING_UNAVAILABLE.")
        out_dir = Path(args.out) / args.version
        out_dir.mkdir(parents=True, exist_ok=True)
        export_fn(out_dir / "model.onnx")
        metrics = {
            "kind": "policy",
            "model": args.task, "version": args.version,
            "algorithm": algorithm,
            "algorithm_rationale": {
                "linucb-contextual-bandit":
                    "single-step decision, no decision-caused state "
                    "transition; linear payoffs, offline logged-bandit fit",
                "cql-h-conservative-q-learning":
                    "sequential allocation coupling via berth occupancy; "
                    "conservative penalty blocks extrapolation to unseen "
                    "actions; gamma=0 (no verified next-state join in logs)",
            }[algorithm],
            "n_actions": replay.n_actions,
            "features": replay.feature_names,
            "output_contract": "score is the recommended action index "
                               "(float); serving is SHADOW mode only",
            "data_source": replay.data_source,
            "ope": report.to_dict(),
            **{f"train.{k}": v for k, v in extra.items()},
        }
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        track.log_artifact(out_dir / "model.onnx")
        print(f"[done] {args.task} {args.version}: {report.reason} -> {out_dir}")
        return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument("--out", default=None,
                        help="default models/<task>")
    parser.add_argument("--version", default="0.1.0")
    parser.add_argument("--min-samples", type=int, default=500)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--min-delta", type=float, default=0.0,
                        help="OPE DR margin over logged baseline required "
                             "for promotion")
    parser.add_argument("--alpha-ucb", type=float, default=1.0)
    parser.add_argument("--alpha-cql", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.0,
                        help="0.0 unless a verified next-state join exists")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.out is None:
        args.out = f"models/{args.task}"
    train(args)


if __name__ == "__main__":
    main()
