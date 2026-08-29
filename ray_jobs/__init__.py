"""Ray distributed compute jobs.

Wraps the trainers for distributed hyperparameter search (Ray Tune) and
distributed data prep. Runs on a local Ray cluster (ray up ray_cluster.yaml)
or k8s via KubeRay. Degrades gracefully: when Ray is not installed or no
cluster is reachable (RAY_UNAVAILABLE) every entry point falls back to local
single-node execution — never fails silently, always announces the mode.
"""
