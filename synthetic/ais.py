"""SYNTHETIC vessel AIS movement generator for Nigerian waters.

Generates trajectories around Lagos/Apapa, Tin Can, Onne, Calabar and the
Nigerian EEZ with embedded anomaly patterns and ground-truth labels:

- loitering: prolonged low-speed drifting outside port approaches
- dark-gap spoofing: AIS transmission gaps followed by implausible jumps
- EEZ incursion: entry into a protected/EEZ polygon without a port call

DATA IS SYNTHETIC. Coordinates are approximate and carry no operational value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import AisConfig

# Approximate port anchorages (lat, lon) — SYNTHETIC-use only.
PORT_ANCHOR = {
    "NGAPP": (6.43, 3.39),   # Lagos Apapa
    "NGTIN": (6.44, 3.35),   # Tin Can
    "NGONN": (4.70, 7.16),   # Onne
    "NGCBQ": (4.98, 8.32),   # Calabar
}
# Simple bounding polygon standing in for a protected EEZ zone (synthetic).
EEZ_BOX = (5.2, 5.9, 3.6, 4.4)  # lat_min, lat_max, lon_min, lon_max


def _in_eez(lat: float, lon: float) -> bool:
    return EEZ_BOX[0] <= lat <= EEZ_BOX[1] and EEZ_BOX[2] <= lon <= EEZ_BOX[3]


def generate_ais(cfg: AisConfig = AisConfig()) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    rows = []
    minutes = cfg.days * 24 * 60
    step = max(5, int(60 / cfg.pings_per_hour))
    ports = list(PORT_ANCHOR)

    for v in range(cfg.n_vessels):
        imo = 9_000_000 + cfg.seed % 100_000 + v
        has_dark = rng.random() < cfg.dark_gap_rate
        has_loiter = rng.random() < cfg.loiter_rate
        has_eez = rng.random() < cfg.eez_incursion_rate

        lat, lon = PORT_ANCHOR[ports[rng.integers(0, len(ports))]]
        t = 0
        dark_until = -1
        dark_start = int(rng.integers(minutes // 4, 3 * minutes // 4)) if has_dark else -1
        loiter_start = int(rng.integers(0, minutes - 720)) if has_loiter else -1

        while t < minutes:
            underway = True
            if loiter_start <= t < loiter_start + 360:  # 6h loiter
                sog = rng.uniform(0.1, 1.2)
                lat += rng.normal(0, 0.004)
                lon += rng.normal(0, 0.004)
            else:
                sog = rng.uniform(8.0, 14.0)
                # drift between ports / offshore
                tgt = PORT_ANCHOR[ports[rng.integers(0, len(ports))]]
                dlat, dlon = tgt[0] - lat, tgt[1] - lon
                norm = max(1e-6, (dlat**2 + dlon**2) ** 0.5)
                lat += dlat / norm * 0.02 + rng.normal(0, 0.003)
                lon += dlon / norm * 0.02 + rng.normal(0, 0.003)

            if has_eez and loiter_start // 2 <= t < loiter_start // 2 + 240:
                # wander into the EEZ box
                lat = EEZ_BOX[0] + rng.uniform(0.05, EEZ_BOX[1] - EEZ_BOX[0] - 0.05)
                lon = EEZ_BOX[2] + rng.uniform(0.05, EEZ_BOX[3] - EEZ_BOX[2] - 0.05)

            if has_dark and t == dark_start:
                dark_until = t + int(rng.integers(360, 1440))  # 6-24h silent

            if dark_until >= t > dark_start:
                t += step
                continue  # transmission gap: no pings

            jumped = False
            if has_dark and t >= dark_until > 0 and t - dark_until < step * 2:
                # reappear implausibly far away (spoof pattern)
                lat += rng.uniform(-0.8, 0.8)
                lon += rng.uniform(-0.8, 0.8)
                jumped = True

            label = "normal"
            if loiter_start <= t < loiter_start + 360:
                label = "loitering"
            if has_eez and _in_eez(lat, lon):
                label = "eez_incursion"
            if jumped:
                label = "dark_gap_reappear"

            rows.append((imo, t, round(lat, 5), round(lon, 5), round(sog, 2),
                         int(_in_eez(lat, lon)), label))
            t += step

    df = pd.DataFrame(rows, columns=[
        "imo", "minute_offset", "lat", "lon", "sog_kn", "in_eez", "movement_label"])
    df["data_source"] = "SYNTHETIC"
    return df


AIS_FEATURES = ["lat", "lon", "sog_kn", "in_eez", "speed_z", "hour_of_day"]


def to_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Feature matrix + binary anomaly label for the anomaly trainers."""
    x = df.copy()
    x["speed_z"] = (x["sog_kn"] - x["sog_kn"].mean()) / (x["sog_kn"].std() + 1e-9)
    x["hour_of_day"] = (x["minute_offset"] // 60) % 24
    y = (x["movement_label"] != "normal").astype(np.int64).to_numpy()
    return x[AIS_FEATURES].to_numpy(dtype=np.float32), y
