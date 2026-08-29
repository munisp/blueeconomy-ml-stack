"""Honest scoring-metrics logging for the inference service.

Appends one JSON line per scoring decision (including SCORING_UNAVAILABLE
events — unavailability is a first-class operational metric, never hidden).
A deploy-time shipper (fluent bit / vector) forwards this file; aggregation
is a platform concern.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

LOG_PATH = Path(os.environ.get("BEML_SCORING_LOG", "results/scoring_log.jsonl"))
_LOCK = threading.Lock()


def log_decision(result: dict, request_id: str = "") -> None:
    record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "request_id": request_id, **result}
    with _LOCK:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")


def availability_rate(path: str | Path = LOG_PATH) -> dict:
    """Fraction of decisions that produced a real score (honest SLO input)."""
    total = ok = 0
    p = Path(path)
    if p.is_file():
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            total += 1
            ok += int(json.loads(line).get("status") == "OK")
    return {"decisions": total, "scored": ok,
            "availability": (ok / total) if total else None}
