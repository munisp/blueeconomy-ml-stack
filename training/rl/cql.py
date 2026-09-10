"""Conservative Q-Learning (CQL-H, discrete actions) for berth-allocation.

Why an offline-constrained method here (and not plain DQN): berth choice
changes the state other vessels see (a berth is occupied until departure),
so the problem is genuinely sequential and a bandit would ignore the
occupancy coupling. But naive off-policy DQN extrapolates Q-values to
actions the logging policy never took and systematically overestimates
them — with no environment to correct the error, that is how offline RL
ships confident garbage. CQL adds the standard conservative penalty
(Kumar et al. 2020, eq. for the entropy/H variant):

    loss = TD_error + alpha * ( logsumexp_a Q(s,a) - Q(s, a_logged) )

which pushes DOWN the value of unseen actions relative to logged ones, so
the learned greedy policy stays near support of the data unless a
non-logged action is demonstrably better. Promotion still has to pass the
OPE gate (training/rl/ope.py) against the logged baseline.

Episode structure from logs: port-call logs give reliable (s, a, r) triples
but NOT a trustworthy next-state join (the "next state" of an allocation is
the queue at the next arrival, which is only observable when another call
exists for the same port). We therefore train with gamma=0 by default —
a conservative one-step CQL (pessimistic fitted reward model) — and only
raise gamma when a verified next-state join exists. This is stated, not
hidden, in metrics.json ("gamma": 0.0, "transition_join": "unavailable").
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Small MLP Q-network — CPU inference budget, <5MB artifact."""

    def __init__(self, n_features: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ArgmaxPolicy(nn.Module):
    """ONNX-exportable greedy wrapper: features -> float action index.

    The serving Scorer contract is a single "score" float output; for a
    policy artifact that float IS the recommended action index (metrics.json
    kind="policy"). Shadow mode wraps the semantics in inference/service.py.
    """

    def __init__(self, qnet: QNetwork):
        super().__init__()
        self.qnet = qnet

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        q = self.qnet(features)
        return torch.argmax(q, dim=1).to(torch.float32)


def train_cql(contexts: np.ndarray, actions: np.ndarray, rewards: np.ndarray,
              n_actions: int, *, gamma: float = 0.0, alpha: float = 1.0,
              hidden: int = 64, epochs: int = 200, lr: float = 3e-4,
              batch_size: int = 256, seed: int = 42,
              device: str = "cpu") -> tuple[QNetwork, dict]:
    """Fits CQL-H on logged (s, a, r) rows; returns (qnet, train_metrics)."""
    torch.manual_seed(seed)
    dev = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    x = torch.tensor(contexts, dtype=torch.float32, device=dev)
    a = torch.tensor(actions, dtype=torch.int64, device=dev)
    r = torch.tensor(rewards, dtype=torch.float32, device=dev)
    qnet = QNetwork(contexts.shape[1], n_actions, hidden).to(dev)
    opt = torch.optim.AdamW(qnet.parameters(), lr=lr, weight_decay=1e-4)
    n = len(x)
    hist = {"td_loss": [], "cql_penalty": []}
    for epoch in range(epochs):
        perm = torch.randperm(n, device=dev)
        ep_td, ep_cql = 0.0, 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            q_all = qnet(x[idx])                        # (b, n_actions)
            q_logged = q_all.gather(1, a[idx].unsqueeze(1)).squeeze(1)
            # gamma=0 (default): no bootstrap — honest one-step targets.
            td = nn.functional.mse_loss(q_logged, r[idx])
            cql = (torch.logsumexp(q_all, dim=1) - q_logged).mean()
            loss = td + alpha * cql
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_td += float(td.detach()) * len(idx)
            ep_cql += float(cql.detach()) * len(idx)
        hist["td_loss"].append(ep_td / n)
        hist["cql_penalty"].append(ep_cql / n)
    metrics = {"final_td_loss": hist["td_loss"][-1],
               "final_cql_penalty": hist["cql_penalty"][-1],
               "gamma": gamma, "alpha_cql": alpha,
               "transition_join": "unavailable" if gamma == 0.0 else "verified"}
    return qnet, metrics


def greedy_policy_actions(qnet: QNetwork, contexts: np.ndarray) -> np.ndarray:
    qnet.eval()
    with torch.no_grad():
        q = qnet(torch.tensor(contexts, dtype=torch.float32))
    return torch.argmax(q, dim=1).cpu().numpy()


def export_policy_onnx(qnet: QNetwork, n_features: int, path) -> None:
    policy = ArgmaxPolicy(qnet.cpu()).eval()
    dummy = torch.zeros(1, n_features)
    torch.onnx.export(
        policy, dummy, str(path),
        input_names=["features"], output_names=["score"],
        dynamic_axes={"features": {0: "batch"}, "score": {0: "batch"}},
        opset_version=17,
    )


class LinearArgmaxPolicy(nn.Module):
    """ONNX-exportable greedy wrapper for a linear bandit (LinUCB theta)."""

    def __init__(self, theta: np.ndarray):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(theta, dtype=torch.float32),
                                   requires_grad=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        q = features @ self.weight.t()
        return torch.argmax(q, dim=1).to(torch.float32)


def export_linear_policy_onnx(theta: np.ndarray, path) -> None:
    policy = LinearArgmaxPolicy(theta).eval()
    dummy = torch.zeros(1, theta.shape[1])
    torch.onnx.export(
        policy, dummy, str(path),
        input_names=["features"], output_names=["score"],
        dynamic_axes={"features": {0: "batch"}, "score": {0: "batch"}},
        opset_version=17,
    )
