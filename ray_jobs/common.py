"""Ray availability handling: graceful degradation to local execution."""

from __future__ import annotations

try:
    import ray
    _HAS_RAY = True
except ImportError:  # pragma: no cover
    ray = None
    _HAS_RAY = False


class RayUnavailable(RuntimeError):
    pass


def init_ray(address: str = "auto", fallback_ok: bool = True) -> bool:
    """Initialise Ray; return True if a cluster is live, False if local.

    RAY_UNAVAILABLE is announced loudly and (with fallback_ok) execution
    continues single-node — the doctrine is degrade, never fabricate.
    """
    if not _HAS_RAY:
        print("[RAY_UNAVAILABLE] ray not installed; falling back to local single-node")
        if fallback_ok:
            return False
        raise RayUnavailable("ray not installed")
    try:
        ray.init(address=address, ignore_reinit_error=True,
                 log_to_driver=False, num_cpus=None)
        print(f"[ray] connected: {ray.cluster_resources()}")
        return True
    except Exception as exc:
        print(f"[RAY_UNAVAILABLE] {exc}; falling back to local single-node")
        if fallback_ok:
            return False
        raise RayUnavailable(str(exc)) from exc
