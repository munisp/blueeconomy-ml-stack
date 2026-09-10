"""Off-policy evaluation (OPE): the promotion gate for RL policies.

A candidate policy is promoted ONLY if its doubly-robust (DR) value estimate
on held-out logged data beats the logged baseline policy's mean reward by at
least min_delta. IPS is reported alongside DR; DR is the gate because it is
consistent when either the behavior-propensity model OR the reward model is
right, and lower-variance than IPS under partial support.

Behavior propensities are NOT logged by the production rules engine, so they
are estimated from the logged (context, action) pairs with a multinomial
logistic model, clipped at eps (documented, not hidden: "propensity_source"
is written into metrics.json). If the propensity model has effectively no
support for the candidate's actions (max weight ratio huge), the gate
refuses promotion honestly rather than trusting an exploded estimate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OPEReport:
    n: int
    baseline_mean_reward: float
    ips_candidate: float
    dr_candidate: float
    match_rate: float               # P(candidate action == logged action)
    max_importance_weight: float
    propensity_source: str
    gate_min_delta: float
    promoted: bool
    reason: str

    def to_dict(self) -> dict:
        return {
            "n_eval": self.n,
            "baseline_mean_reward": self.baseline_mean_reward,
            "ips_candidate": self.ips_candidate,
            "dr_candidate": self.dr_candidate,
            "match_rate": self.match_rate,
            "max_importance_weight": self.max_importance_weight,
            "propensity_source": self.propensity_source,
            "gate_min_delta": self.gate_min_delta,
            "promoted": self.promoted,
            "gate_reason": self.reason,
        }


def estimate_behavior_propensities(contexts: np.ndarray, actions: np.ndarray,
                                   n_actions: int, eps: float = 0.05) -> np.ndarray:
    """p_b(a|x) via multinomial logistic regression, clipped to [eps, 1]."""
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(contexts, actions)
    proba = clf.predict_proba(contexts)
    full = np.full((len(contexts), n_actions), eps, dtype=np.float64)
    for j, cls in enumerate(clf.classes_):
        full[:, int(cls)] = proba[:, j]
    return np.clip(full, eps, 1.0)


def fit_reward_model(contexts: np.ndarray, actions: np.ndarray,
                     rewards: np.ndarray, n_actions: int) -> "object":
    """r_hat(x, a): ridge regression on [x, onehot(a)] features."""
    from sklearn.linear_model import Ridge
    onehot = np.eye(n_actions, dtype=np.float64)[actions]
    feats = np.concatenate([contexts, onehot], axis=1)
    model = Ridge(alpha=1.0)
    model.fit(feats, rewards)

    def predict(ctx: np.ndarray, act: np.ndarray) -> np.ndarray:
        oh = np.eye(n_actions, dtype=np.float64)[act]
        return model.predict(np.concatenate([ctx, oh], axis=1))

    return predict


def evaluate(candidate_actions: np.ndarray, contexts: np.ndarray,
             actions: np.ndarray, rewards: np.ndarray, n_actions: int,
             *, min_delta: float = 0.0, eps: float = 0.05,
             max_weight: float = 100.0, min_match_rate: float = 0.05,
             reward_model=None) -> OPEReport:
    """IPS + DR evaluation of a deterministic candidate policy.

    candidate_actions: pi(x) for each logged row (same ordering).
    Returns an OPEReport whose .promoted is the gate decision.
    """
    n = len(rewards)
    baseline = float(np.mean(rewards))
    p_b = estimate_behavior_propensities(contexts, actions, n_actions, eps)
    p_logged = p_b[np.arange(n), actions]
    p_cand = p_b[np.arange(n), candidate_actions]
    # Actions the logging policy effectively never took sit at the clip
    # floor; a candidate that leans on them is off-support no matter what
    # the reward model claims.
    floor_frac = float(np.mean(p_cand <= eps + 1e-12))
    match = (candidate_actions == actions).astype(np.float64)
    w = match / p_logged
    ips = float(np.mean(w * rewards))
    if reward_model is None:
        reward_model = fit_reward_model(contexts, actions, rewards, n_actions)
    r_hat_logged = reward_model(contexts, actions)
    r_hat_cand = reward_model(contexts, candidate_actions)
    dr = float(np.mean(r_hat_cand + w * (rewards - r_hat_logged)))
    max_w = float(np.max(1.0 / p_logged)) if n else float("inf")
    match_rate = float(np.mean(match))
    if match_rate < min_match_rate:
        # The candidate almost never agrees with the logging policy: there
        # is effectively NO logged support for its actions, so both IPS and
        # DR reduce to pure reward-model extrapolation. Honest refusal —
        # offline OPE cannot vouch for a policy that far off support.
        return OPEReport(n, baseline, ips, dr, match_rate, max_w,
                         "estimated:logistic-regression(eps=%.2f)" % eps,
                         min_delta, False,
                         f"REJECT: overlap failure — candidate matches the "
                         f"logged policy on {match_rate:.1%} of rows "
                         f"(< {min_match_rate:.0%}); OPE would be reward-model "
                         f"extrapolation, promotion refused")
    if floor_frac > min_match_rate:
        return OPEReport(n, baseline, ips, dr, match_rate, max_w,
                         "estimated:logistic-regression(eps=%.2f)" % eps,
                         min_delta, False,
                         f"REJECT: support failure — candidate takes actions "
                         f"at/below the propensity floor on {floor_frac:.1%} "
                         f"of rows (actions essentially never logged); "
                         f"promotion refused")
    if max_w > max_weight:
        return OPEReport(n, baseline, ips, dr, match_rate, max_w,
                         "estimated:logistic-regression(eps=%.2f)" % eps,
                         min_delta, False,
                         f"REJECT: propensity support too thin "
                         f"(max importance weight {max_w:.1f} > {max_weight}); "
                         f"OPE would be extrapolation, promotion refused")
    promoted = dr > baseline + min_delta
    reason = ("PROMOTE: DR estimate %.4f beats logged baseline %.4f + delta %.4f"
              % (dr, baseline, min_delta)) if promoted else (
        "REJECT: DR estimate %.4f does not beat logged baseline %.4f + delta %.4f"
        % (dr, baseline, min_delta))
    return OPEReport(n, baseline, ips, dr, match_rate, max_w,
                     "estimated:logistic-regression(eps=%.2f)" % eps,
                     min_delta, promoted, reason)
