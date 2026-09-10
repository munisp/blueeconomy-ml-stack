"""Offline RL / contextual bandit training framework (Phase 18).

Doctrine: OFFLINE learning from REAL logged platform data only, gated by
config DSNs; never a synthetic training default. Promotion is gated by
off-policy evaluation (OPE) against the logged baseline policy — a candidate
policy that does not beat the baseline on the OPE estimate is NEVER exported,
and the serving registry keeps reporting SCORING_UNAVAILABLE (fail-closed).
Synthetic data exists in tests/ only.

Modules:
- data:     replay dataset builders over real logged tables (config-gated DSNs)
- bandits:  LinUCB contextual bandit (queue-policy, route-advice)
- cql:      conservative Q-learning (CQL-H) for berth-allocation
- ope:      inverse-propensity scoring (IPS) and doubly-robust (DR) evaluation
- train:    CLI entrypoint per task with the eval-gated ONNX export
"""
