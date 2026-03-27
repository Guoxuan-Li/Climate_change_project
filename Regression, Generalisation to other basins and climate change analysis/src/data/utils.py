"""
Data utilities: normalization helpers, direction label mappings, feature extraction.
"""
import numpy as np
import torch

from src.config import (
    DIRECTION_LABELS, INTENSITY_LABELS, NUM_DIRECTION_CLASSES,
    ENV_FEATURE_DIM, DATA1D_COLS, DATA1D_FEATURE_COLS,
    NORM_LONG, NORM_LAT, NORM_PRES, NORM_WND,
)


# ── Direction helpers ──────────────────────────────────────────────────────

# Angle midpoints in radians (counterclockwise from east), matching env_data.py
DIRECTION_ANGLES_RAD = np.array([
    0,              # 0: E
    np.pi / 4,      # 1: NE
    np.pi / 2,      # 2: N
    3 * np.pi / 4,  # 3: NW
    np.pi,          # 4: W
    5 * np.pi / 4,  # 5: SW
    3 * np.pi / 2,  # 6: S
    7 * np.pi / 4,  # 7: SE
])


def direction_to_displacement(direction_class: int, speed_km_per_6h: float,
                              hours: float = 24.0) -> tuple[float, float]:
    """Convert a direction class + speed into (dx_km, dy_km) displacement.

    Args:
        direction_class: int 0-7
        speed_km_per_6h: movement speed in km per 6h interval
        hours: prediction horizon (default 24h = 4 intervals)

    Returns:
        (dx_km, dy_km): eastward and northward displacements in km
    """
    angle = DIRECTION_ANGLES_RAD[direction_class]
    total_dist = speed_km_per_6h * (hours / 6.0)
    dx = total_dist * np.cos(angle)
    dy = total_dist * np.sin(angle)
    return dx, dy


# ── Env-Data feature extraction ────────────────────────────────────────────

def env_dict_to_vector(env_dict: dict) -> np.ndarray:
    """Flatten an Env-Data dict into a fixed-size feature vector (92 dims).

    Features:
        area (6) + wind (1) + intensity_class (6) + move_velocity (1) +
        month (12) + location_long (36) + location_lat (12) +
        history_direction12 (8) + history_direction24 (8) +
        missing_hist12 (1) + missing_hist24 (1) = 92
    """
    parts = []

    # area: one-hot (6,)
    parts.append(np.asarray(env_dict["area"], dtype=np.float32))

    # wind: scalar
    parts.append(np.array([float(env_dict["wind"])], dtype=np.float32))

    # intensity_class: one-hot (6,)
    parts.append(np.asarray(env_dict["intensity_class"], dtype=np.float32))

    # move_velocity: scalar
    parts.append(np.array([float(env_dict["move_velocity"])], dtype=np.float32))

    # month: one-hot (12,)
    parts.append(np.asarray(env_dict["month"], dtype=np.float32))

    # location_long: one-hot (36,)
    parts.append(np.asarray(env_dict["location_long"], dtype=np.float32))

    # location_lat: one-hot (12,)
    parts.append(np.asarray(env_dict["location_lat"], dtype=np.float32))

    # history_direction12: one-hot (8,) or -1 (missing)
    hd12 = env_dict["history_direction12"]
    if isinstance(hd12, (int, float)) and hd12 == -1:
        parts.append(np.zeros(8, dtype=np.float32))
        missing_12 = 1.0
    else:
        parts.append(np.asarray(hd12, dtype=np.float32))
        missing_12 = 0.0

    # history_direction24: one-hot (8,) or -1 (missing)
    hd24 = env_dict["history_direction24"]
    if isinstance(hd24, (int, float)) and hd24 == -1:
        parts.append(np.zeros(8, dtype=np.float32))
        missing_24 = 1.0
    else:
        parts.append(np.asarray(hd24, dtype=np.float32))
        missing_24 = 0.0

    # Missing-history flags
    parts.append(np.array([missing_12, missing_24], dtype=np.float32))

    vec = np.concatenate(parts)
    assert vec.shape == (ENV_FEATURE_DIM,), f"Expected {ENV_FEATURE_DIM}, got {vec.shape}"
    return vec


# ── Data1D parsing ─────────────────────────────────────────────────────────

def parse_data1d_file(path: str) -> list[dict]:
    """Parse a Data1D TSV file into a list of timestep dicts.

    Returns list of dicts with keys: id, flag, long_norm, lat_norm,
    pres_norm, wnd_norm, timestamp, name
    """
    import pandas as pd
    df = pd.read_csv(path, delimiter='\t', header=None, names=DATA1D_COLS)
    # Ensure timestamp is string
    df["timestamp"] = df["timestamp"].astype(str).str.strip()
    return df.to_dict("records")


# ── Class weights computation ──────────────────────────────────────────────

def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """Compute inverse-frequency class weights for imbalanced classification.

    Returns a float tensor of shape (num_classes,).
    """
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)  # avoid division by zero
    weights = 1.0 / (counts * num_classes)
    weights = weights / weights.sum() * num_classes  # normalize so mean = 1
    return torch.tensor(weights, dtype=torch.float32)
