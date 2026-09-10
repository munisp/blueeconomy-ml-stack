"""LinUCB contextual bandit — chosen for queue-policy and route-advice.

Why a contextual bandit here (and not full RL): these decisions are
single-step — a priority bucket or route option is picked once per entity,
the reward is observed, and there is no meaningful state transition the
decision influences (the next declaration's queue state is driven by
arrivals, not by this pick). A bandit is the honest model of that problem;
a sequential RL method would invent transition structure the data does not
contain.

Why LinUCB specifically: linear-in-features payoffs with per-arm confidence
sets; it trains offline from logged (context, action, reward) triples by
updating only the arm that was actually pulled (standard logged-bandit
learning), is deterministic given the data, runs on CPU in milliseconds,
and exports cleanly to ONNX as a linear argmax policy.

The trained SERVING policy is the greedy mean-reward policy (no exploration
on production — shadow mode only, see inference/service.py).
"""

from __future__ import annotations

import numpy as np


class LinUCB:
    """Disjoint LinUCB: one ridge regression per arm (A_a x ≈ b_a)."""

    def __init__(self, n_actions: int, n_features: int, alpha: float = 1.0,
                 ridge: float = 1.0):
        self.n_actions, self.n_features = n_actions, n_features
        self.alpha = alpha
        self.A = np.stack([np.eye(n_features) * ridge for _ in range(n_actions)])
        self.b = np.zeros((n_actions, n_features))

    def update(self, x: np.ndarray, action: int, reward: float) -> None:
        self.A[action] += np.outer(x, x)
        self.b[action] += reward * x

    def fit(self, contexts: np.ndarray, actions: np.ndarray,
            rewards: np.ndarray) -> "LinUCB":
        for x, a, r in zip(contexts, actions, rewards):
            self.update(np.asarray(x, dtype=np.float64), int(a), float(r))
        return self

    def theta(self) -> np.ndarray:
        """Per-arm weight matrix (n_actions, n_features)."""
        return np.stack([np.linalg.solve(self.A[a], self.b[a])
                         for a in range(self.n_actions)])

    def mean_rewards(self, contexts: np.ndarray) -> np.ndarray:
        return contexts @ self.theta().T

    def ucb_scores(self, contexts: np.ndarray) -> np.ndarray:
        th = self.theta()
        means = contexts @ th.T
        out = np.empty_like(means)
        for a in range(self.n_actions):
            a_inv = np.linalg.inv(self.A[a])
            bonus = np.sqrt(np.einsum("ni,ij,nj->n", contexts, a_inv, contexts))
            out[:, a] = means[:, a] + self.alpha * bonus
        return out

    def greedy_actions(self, contexts: np.ndarray) -> np.ndarray:
        return np.argmax(self.mean_rewards(contexts), axis=1)
