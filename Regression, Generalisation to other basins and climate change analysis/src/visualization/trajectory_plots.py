"""
Trajectory visualization: actual vs predicted cyclone tracks on geographic maps.

Provides functions for:
- Single-storm track comparison (actual vs one or more predicted tracks)
- Multi-storm grid overview
- Along-track error accumulation plots
- End-to-end test set visualization from checkpoints

All geographic plots use Cartopy PlateCarree projection with a dark-background
style inspired by the original TropiCycloneNet starter notebook.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import torch

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

from src.config import (
    DATA_ROOT, DATA1D_ROOT, DATA1D_COLS, DATA1D_FEATURE_COLS,
    MASTER_INDEX_PATH, SEQ_LEN, NUM_DIRECTION_CLASSES,
    NORM_LONG, NORM_LAT, NORM_WND, VELOCITY_NORM_FACTOR,
    DIRECTION_LABELS, ENV_FEATURE_DIM, HIDDEN_DIM, PROJECT_ROOT,
)
from src.data.utils import DIRECTION_ANGLES_RAD, env_dict_to_vector

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Model name -> (checkpoint glob pattern, model class path, model_type, is_regression)
# model_type: "env_single", "data1d_seq", "env_seq", "data3d", "fusion"
MODEL_REGISTRY = {
    # Classification models
    "Cls 1: Env MLP": {
        "ckpt_pattern": "stage1_env_mlp_best.pt",
        "model_cls": "src.models.baseline_mlp.BaselineMLP",
        "model_type": "env_single",
        "is_regression": False,
    },
    "Cls 2: LSTM": {
        "ckpt_pattern": "stage2_lstm_1d_best.pt",
        "model_cls": "src.models.lstm_1d.LSTMTracker",
        "model_type": "data1d_seq",
        "is_regression": False,
    },
    "Cls 3: Env Temporal": {
        "ckpt_pattern": "stage3_env_temporal_best.pt",
        "model_cls": "src.models.env_temporal.EnvTemporalModel",
        "model_type": "env_seq",
        "is_regression": False,
    },
    "Cls 4: CNN": {
        "ckpt_pattern": "stage4_cnn_3d_best.pt",
        "model_cls": "src.models.cnn_3d.CNNEncoder3D",
        "model_type": "data3d",
        "is_regression": False,
    },
    "Cls 5: Fusion": {
        "ckpt_pattern": "stage5_fusion_best.pt",
        "model_cls": "src.models.fusion_model.FusionModel",
        "model_type": "fusion",
        "is_regression": False,
    },
    # Regression models
    "Reg 1: Env MLP": {
        "ckpt_pattern": "reg_stage1_env_mlp_best.pt",
        "model_cls": "src.models.regression_models.RegMLP",
        "model_type": "env_single",
        "is_regression": True,
    },
    "Reg 2: LSTM": {
        "ckpt_pattern": "reg_stage2_lstm_1d_best.pt",
        "model_cls": "src.models.regression_models.RegLSTM",
        "model_type": "data1d_seq",
        "is_regression": True,
    },
    "Reg 3: Env Temporal": {
        "ckpt_pattern": "reg_stage3_env_temporal_best.pt",
        "model_cls": "src.models.regression_models.RegEnvTemporal",
        "model_type": "env_seq",
        "is_regression": True,
    },
    "Reg 4: CNN": {
        "ckpt_pattern": "reg_stage4_cnn_3d_best.pt",
        "model_cls": "src.models.regression_models.RegCNN3D",
        "model_type": "data3d",
        "is_regression": True,
    },
    "Reg 5: Fusion": {
        "ckpt_pattern": "reg_stage5_fusion_best.pt",
        "model_cls": "src.models.regression_models.RegFusionModel",
        "model_type": "fusion",
        "is_regression": True,
    },
}

# Palette for predicted tracks (colour-blind friendly, high contrast on dark bg)
PRED_COLORS = [
    "#FF6B6B",   # coral red
    "#4ECDC4",   # teal
    "#FFE66D",   # gold
    "#A8E6CF",   # mint
    "#FF8B94",   # salmon
    "#B8A9C9",   # lavender
]

# Wind-speed Saffir-Simpson-like colour boundaries (m/s) for the track colourmap
_WIND_BOUNDS = [0, 17, 25, 33, 43, 50, 58, 70, 100]
_WIND_COLORS = [
    "#5EBAFF",   # TD
    "#00FAF4",   # TS
    "#FFFFCC",   # C1
    "#FFE775",   # C2
    "#FFC140",   # C3
    "#FF8F20",   # C4
    "#FF6060",   # C5
    "#C00000",   # C5+
]

HOURS_PER_STEP = 6   # Data1D temporal resolution


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def norm_to_lon(long_norm: float) -> float:
    """Convert normalised longitude to degrees East."""
    return (long_norm * NORM_LONG["scale"] + NORM_LONG["offset"]) / 10.0


def norm_to_lat(lat_norm: float) -> float:
    """Convert normalised latitude to degrees North."""
    return (lat_norm * NORM_LAT["scale"] + NORM_LAT["offset"]) / 10.0


def norm_to_wind(wnd_norm: float) -> float:
    """Convert normalised wind to m/s."""
    return wnd_norm * NORM_WND["scale"] + NORM_WND["offset"]


def _wind_colormap():
    """Build a ListedColormap matching TC intensity categories."""
    cmap = mcolors.ListedColormap(_WIND_COLORS)
    norm = mcolors.BoundaryNorm(_WIND_BOUNDS, cmap.N)
    return cmap, norm


def haversine_km(lon1, lat1, lon2, lat2):
    """Great-circle distance between two points in km."""
    R = 6371.0
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def km_to_deg(dx_km, dy_km, lat_ref):
    """Convert km displacements to approximate degree displacements."""
    d_lat = dy_km / 111.32
    d_lon = dx_km / (111.32 * np.cos(np.radians(lat_ref)))
    return d_lon, d_lat


def _load_data3d_tensor(data_root, basin, year, storm_name, timestamp):
    """Load a single Data3D NetCDF into a (13, 81, 81) tensor for CNN inference."""
    nc_path = (Path(data_root) / "Data3D" / basin / str(year) / storm_name /
               f"TCND_{storm_name}_{timestamp}_sst_z_u_v.nc")
    if not nc_path.exists():
        return None
    import xarray as xr
    ds = xr.open_dataset(str(nc_path))
    u = ds["u"].values[0]  # (4, 81, 81)
    v = ds["v"].values[0]
    z = ds["z"].values[0] / 10000.0
    sst_raw = ds["sst"].values
    sst = np.where(np.isfinite(sst_raw) & (np.abs(sst_raw) < 1e10), sst_raw, np.nan)
    sst = np.nan_to_num(sst, nan=0.0)
    sst = (sst - 290.0) / 20.0
    ds.close()
    channels = np.concatenate([u, v, z, sst[np.newaxis]], axis=0)
    return torch.tensor(channels, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Dark-background map styling context manager
# ---------------------------------------------------------------------------

class _DarkMapStyle:
    """Context manager that applies dark-background styling and restores after."""

    _OCEAN_COLOR = "#1a1a2e"
    _LAND_COLOR = "#16213e"
    _COAST_COLOR = "#e0e0e0"
    _GRID_COLOR = "#555555"

    def __enter__(self):
        self._prev_style = matplotlib.rcParams.copy()
        plt.style.use("dark_background")
        return self

    def __exit__(self, *exc):
        matplotlib.rcParams.update(self._prev_style)
        return False

    @staticmethod
    def dress_ax(ax):
        """Add land, ocean, coastlines, gridlines to a Cartopy GeoAxes."""
        ax.set_facecolor(_DarkMapStyle._OCEAN_COLOR)
        ax.add_feature(
            cfeature.LAND, facecolor=_DarkMapStyle._LAND_COLOR,
            edgecolor="none", zorder=0,
        )
        ax.add_feature(
            cfeature.COASTLINE, linewidth=0.6,
            edgecolor=_DarkMapStyle._COAST_COLOR, zorder=1,
        )
        gl = ax.gridlines(
            draw_labels=True, linewidth=0.4,
            color=_DarkMapStyle._GRID_COLOR, alpha=0.6,
            linestyle="--",
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {"size": 7, "color": "#cccccc"}
        gl.ylabel_style = {"size": 7, "color": "#cccccc"}
        return gl


# ---------------------------------------------------------------------------
# 1. Single-storm track comparison
# ---------------------------------------------------------------------------

def plot_storm_track_comparison(
    storm_name: str,
    year: int,
    actual_track: np.ndarray,
    predicted_tracks: dict[str, np.ndarray],
    output_path: str | Path,
    wind_speeds: Optional[np.ndarray] = None,
):
    """Plot a single storm's actual vs predicted trajectory on a geographic map.

    Args:
        storm_name: e.g. "HAIYAN"
        year: e.g. 2013
        actual_track: (N, 2) array of (lon, lat) in degrees
        predicted_tracks: dict mapping model_name -> (M, 2) array of (lon, lat)
        output_path: where to save the figure
        wind_speeds: optional (N,) array of wind speeds in m/s for colouring
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with _DarkMapStyle():
        proj = ccrs.PlateCarree()
        fig, ax = plt.subplots(
            figsize=(10, 7), subplot_kw={"projection": proj},
        )
        _DarkMapStyle.dress_ax(ax)

        # Determine map extent from actual track only (predicted can have outliers)
        pad = 5.0
        ax.set_extent([
            actual_track[:, 0].min() - pad, actual_track[:, 0].max() + pad,
            actual_track[:, 1].min() - pad, actual_track[:, 1].max() + pad,
        ], crs=proj)

        # --- Actual track ---
        if wind_speeds is not None and len(wind_speeds) == len(actual_track):
            cmap, norm = _wind_colormap()
            for i in range(len(actual_track) - 1):
                ax.plot(
                    actual_track[i:i + 2, 0], actual_track[i:i + 2, 1],
                    color=cmap(norm(wind_speeds[i])),
                    linewidth=2.2, solid_capstyle="round",
                    transform=proj, zorder=5,
                )
            sc = ax.scatter(
                actual_track[:, 0], actual_track[:, 1],
                c=wind_speeds, cmap=cmap, norm=norm,
                s=28, edgecolors="white", linewidths=0.3,
                transform=proj, zorder=6, label="Actual",
            )
            cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.02, aspect=25)
            cbar.set_label("Wind speed (m/s)", fontsize=9, color="#cccccc")
            cbar.ax.yaxis.set_tick_params(color="#cccccc")
            plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#cccccc", fontsize=8)
        else:
            ax.plot(
                actual_track[:, 0], actual_track[:, 1],
                color="white", linewidth=2.2, solid_capstyle="round",
                transform=proj, zorder=5, label="Actual",
            )
            ax.scatter(
                actual_track[:, 0], actual_track[:, 1],
                color="white", s=28, edgecolors="white", linewidths=0.3,
                transform=proj, zorder=6,
            )

        # Start / end markers for actual track
        ax.plot(
            actual_track[0, 0], actual_track[0, 1],
            marker="^", color="#00FF7F", markersize=12,
            markeredgecolor="white", markeredgewidth=0.8,
            transform=proj, zorder=8,
        )
        ax.plot(
            actual_track[-1, 0], actual_track[-1, 1],
            marker="*", color="#FF4500", markersize=14,
            markeredgecolor="white", markeredgewidth=0.8,
            transform=proj, zorder=8,
        )

        # --- Predicted tracks ---
        for idx, (model_name, pred_track) in enumerate(predicted_tracks.items()):
            color = PRED_COLORS[idx % len(PRED_COLORS)]
            ax.plot(
                pred_track[:, 0], pred_track[:, 1],
                color=color, linewidth=1.0, linestyle="--", alpha=0.85,
                transform=proj, zorder=4, label=model_name,
            )
            ax.scatter(
                pred_track[:, 0], pred_track[:, 1],
                color=color, s=10, marker="D",
                edgecolors="white", linewidths=0.15,
                transform=proj, zorder=4,
            )

        # Title — clean, no error subtitle
        ax.set_title(f"{storm_name} ({year})", fontsize=13, fontweight="bold",
                      color="white", pad=10)

        # Legend
        leg = ax.legend(
            loc="lower left", fontsize=8, framealpha=0.7,
            facecolor="#2a2a2a", edgecolor="#555555",
        )
        for text in leg.get_texts():
            text.set_color("white")

        plt.tight_layout()
        fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)


# ---------------------------------------------------------------------------
# 2. predict_trajectory_classification
# ---------------------------------------------------------------------------

def predict_trajectory_classification(
    model: torch.nn.Module,
    storm_data: dict,
    env_data_list: list[dict],
    device: torch.device,
    model_type: str = "data1d_seq",
) -> np.ndarray:
    """Predict trajectory step-by-step using a classification model.

    At each timestep (with SEQ_LEN of history), predict the direction class,
    then convert to a displacement using the actual movement velocity from
    Env-Data.  Actual positions (not predicted) are used as model input so
    that direction prediction quality is evaluated in isolation.

    Args:
        model: classification model (returns logits of shape (B, 8))
        storm_data: dict with keys:
            "features": (T, 4) normalised Data1D features for the full storm
            "positions": (T, 2) actual (lon, lat) in degrees
        env_data_list: list of T env-data dicts (one per timestep)
        device: torch device
        model_type: one of "data1d_seq", "env_seq", "env_single"

    Returns:
        predicted_positions: (T, 2) array of (lon, lat) in degrees.
        The first SEQ_LEN positions are copied from actual; subsequent ones
        are predicted.
    """
    model.eval()
    features = storm_data["features"]      # (T, 4)
    positions = storm_data["positions"]    # (T, 2)  degrees
    T = len(features)

    predicted = np.copy(positions)  # start with actual

    with torch.no_grad():
        for t in range(SEQ_LEN, T):
            # --- build model input from actual history ---
            if model_type == "data1d_seq":
                seq = features[t - SEQ_LEN:t]         # (8, 4)
                x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
                logits = model(x)                      # (1, 8)

            elif model_type == "env_seq":
                vecs = []
                for j in range(t - SEQ_LEN, t):
                    vecs.append(env_dict_to_vector(env_data_list[j]))
                x = torch.tensor(np.stack(vecs), dtype=torch.float32).unsqueeze(0).to(device)
                logits = model(x)

            elif model_type == "env_single":
                vec = env_dict_to_vector(env_data_list[t])
                x = torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(device)
                logits = model(x)

            elif model_type == "data3d":
                d3d = storm_data.get("data3d_tensors", {}).get(t)
                if d3d is None:
                    predicted[t] = positions[t]  # fallback to actual
                    continue
                x = d3d.unsqueeze(0).to(device)
                logits = model(x)

            elif model_type == "fusion":
                # Fusion needs all 3: data1d_seq, env_seq, data3d
                seq_1d = features[t - SEQ_LEN:t]
                x_1d = torch.tensor(seq_1d, dtype=torch.float32).unsqueeze(0).to(device)

                vecs = []
                for j in range(t - SEQ_LEN, t):
                    vecs.append(env_dict_to_vector(env_data_list[j]))
                x_env = torch.tensor(np.stack(vecs), dtype=torch.float32).unsqueeze(0).to(device)

                d3d = storm_data.get("data3d_tensors", {}).get(t)
                if d3d is None:
                    predicted[t] = positions[t]
                    continue
                x_3d = d3d.unsqueeze(0).to(device)

                logits = model(x_1d, x_env, x_3d)

            else:
                raise ValueError(f"Unknown model_type: {model_type}")

            pred_class = int(logits.argmax(dim=1).item())

            # Movement velocity (normalised) from Env-Data; denormalise to km
            vel_norm = float(env_data_list[t]["move_velocity"])
            speed_km = vel_norm * VELOCITY_NORM_FACTOR  # km per 6h interval

            # Displacement in km for one 6-hour step
            angle = DIRECTION_ANGLES_RAD[pred_class]
            dx_km = speed_km * np.cos(angle)
            dy_km = speed_km * np.sin(angle)

            # Convert to degrees
            ref_lat = positions[t - 1, 1]
            d_lon, d_lat = km_to_deg(dx_km, dy_km, ref_lat)

            # Predicted position = actual previous + predicted delta
            predicted[t, 0] = positions[t - 1, 0] + d_lon
            predicted[t, 1] = positions[t - 1, 1] + d_lat

    return predicted


# ---------------------------------------------------------------------------
# 3. predict_trajectory_regression
# ---------------------------------------------------------------------------

def predict_trajectory_regression(
    model: torch.nn.Module,
    storm_data: dict,
    env_data_list: list[dict],
    device: torch.device,
    model_type: str = "data1d_seq",
) -> np.ndarray:
    """Predict trajectory using a regression model (delta_lon, delta_lat).

    At each timestep, the model predicts normalised (delta_lon, delta_lat).
    Degrees are recovered as delta_deg = delta_norm * 5.0.  Actual history
    is used as input at every step.

    Supports all model types: data1d_seq, env_single, env_seq, data3d, fusion.

    Returns:
        predicted_positions: (T, 2)
    """
    model.eval()
    features = storm_data["features"]
    positions = storm_data["positions"]
    T = len(features)

    predicted = np.copy(positions)

    with torch.no_grad():
        for t in range(SEQ_LEN, T):
            if model_type == "data1d_seq":
                seq = features[t - SEQ_LEN:t]
                x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
                delta_norm = model(x).cpu().numpy().squeeze()

            elif model_type == "env_single":
                vec = env_dict_to_vector(env_data_list[t])
                x = torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(device)
                delta_norm = model(x).cpu().numpy().squeeze()

            elif model_type == "env_seq":
                vecs = [env_dict_to_vector(env_data_list[j])
                        for j in range(t - SEQ_LEN, t)]
                x = torch.tensor(np.stack(vecs), dtype=torch.float32).unsqueeze(0).to(device)
                delta_norm = model(x).cpu().numpy().squeeze()

            elif model_type == "data3d":
                d3d = storm_data.get("data3d_tensors", {}).get(t)
                if d3d is None:
                    predicted[t] = positions[t]
                    continue
                x = d3d.unsqueeze(0).to(device)
                delta_norm = model(x).cpu().numpy().squeeze()

            elif model_type == "fusion":
                seq_1d = features[t - SEQ_LEN:t]
                x_1d = torch.tensor(seq_1d, dtype=torch.float32).unsqueeze(0).to(device)

                vecs = [env_dict_to_vector(env_data_list[j])
                        for j in range(t - SEQ_LEN, t)]
                x_env = torch.tensor(np.stack(vecs), dtype=torch.float32).unsqueeze(0).to(device)

                d3d = storm_data.get("data3d_tensors", {}).get(t)
                if d3d is None:
                    predicted[t] = positions[t]
                    continue
                x_3d = d3d.unsqueeze(0).to(device)

                delta_norm = model(x_1d, x_env, x_3d).cpu().numpy().squeeze()
            else:
                raise ValueError(f"Unknown model_type: {model_type}")

            delta_deg = delta_norm * 5.0
            predicted[t, 0] = positions[t - 1, 0] + delta_deg[0]
            predicted[t, 1] = positions[t - 1, 1] + delta_deg[1]

    return predicted


# ---------------------------------------------------------------------------
# 4. Multi-storm grid
# ---------------------------------------------------------------------------

def plot_multi_storm_grid(
    storms_data: list[dict],
    output_path: str | Path,
    ncols: int = 3,
):
    """Plot a grid of multiple storms' actual vs predicted tracks.

    Args:
        storms_data: list of dicts, each with keys:
            "storm_name", "year", "actual_track" (N,2),
            "predicted_tracks" (dict[str, (M,2)]),
            "wind_speeds" (optional, (N,))
        output_path: where to save the figure
        ncols: number of columns
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(storms_data)
    nrows = math.ceil(n / ncols)

    with _DarkMapStyle():
        proj = ccrs.PlateCarree()
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(5.5 * ncols, 4.5 * nrows),
            subplot_kw={"projection": proj},
        )
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = axes[np.newaxis, :]
        elif ncols == 1:
            axes = axes[:, np.newaxis]

        for idx in range(nrows * ncols):
            r, c = divmod(idx, ncols)
            ax = axes[r, c]

            if idx >= n:
                ax.set_visible(False)
                continue

            sd = storms_data[idx]
            actual = sd["actual_track"]
            preds = sd["predicted_tracks"]
            winds = sd.get("wind_speeds", None)

            _DarkMapStyle.dress_ax(ax)

            # Extent based on actual track only
            pad = 3.0
            ax.set_extent([
                actual[:, 0].min() - pad, actual[:, 0].max() + pad,
                actual[:, 1].min() - pad, actual[:, 1].max() + pad,
            ], crs=proj)

            # Actual track
            if winds is not None and len(winds) == len(actual):
                cmap, norm = _wind_colormap()
                for i in range(len(actual) - 1):
                    ax.plot(
                        actual[i:i + 2, 0], actual[i:i + 2, 1],
                        color=cmap(norm(winds[i])),
                        linewidth=1.8, solid_capstyle="round",
                        transform=proj, zorder=5,
                    )
                ax.scatter(
                    actual[:, 0], actual[:, 1],
                    c=winds, cmap=cmap, norm=norm,
                    s=16, edgecolors="white", linewidths=0.2,
                    transform=proj, zorder=6,
                )
            else:
                ax.plot(
                    actual[:, 0], actual[:, 1],
                    color="white", linewidth=1.8,
                    transform=proj, zorder=5,
                )

            # Start/end
            ax.plot(
                actual[0, 0], actual[0, 1],
                marker="^", color="#00FF7F", markersize=8,
                markeredgecolor="white", markeredgewidth=0.5,
                transform=proj, zorder=8,
            )
            ax.plot(
                actual[-1, 0], actual[-1, 1],
                marker="*", color="#FF4500", markersize=10,
                markeredgecolor="white", markeredgewidth=0.5,
                transform=proj, zorder=8,
            )

            # Predicted tracks
            for pidx, (mname, pred) in enumerate(preds.items()):
                color = PRED_COLORS[pidx % len(PRED_COLORS)]
                ax.plot(
                    pred[:, 0], pred[:, 1],
                    color=color, linewidth=0.8, linestyle="--", alpha=0.85,
                    transform=proj, zorder=4,
                )

            # Clean subplot title — no error numbers
            ax.set_title(f"{sd['storm_name']} ({sd['year']})",
                         fontsize=9, fontweight="bold", color="white", pad=6)

        # Build a single legend for the whole figure
        legend_elements = [
            plt.Line2D([0], [0], color="white", lw=2, label="Actual"),
        ]
        # Use model names from the first storm that has predictions
        if storms_data:
            for pidx, mname in enumerate(storms_data[0]["predicted_tracks"].keys()):
                legend_elements.append(
                    plt.Line2D(
                        [0], [0], color=PRED_COLORS[pidx % len(PRED_COLORS)],
                        lw=1.5, linestyle="--", label=mname,
                    )
                )
        fig.legend(
            handles=legend_elements, loc="lower center",
            ncol=len(legend_elements), fontsize=8,
            framealpha=0.7, facecolor="#2a2a2a", edgecolor="#555555",
            labelcolor="white",
        )

        fig.suptitle(
            "Cyclone Track Comparison — Selected Test Storms",
            fontsize=14, fontweight="bold", color="white", y=1.01,
        )
        plt.tight_layout(rect=[0, 0.04, 1, 0.98])
        fig.savefig(
            output_path, dpi=200, bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
        plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Along-track error plot
# ---------------------------------------------------------------------------

def plot_error_along_track(
    storm_name: str,
    actual_positions: np.ndarray,
    predicted_positions: dict[str, np.ndarray],
    output_path: str | Path,
):
    """Plot how prediction error accumulates along the track.

    Args:
        storm_name: for the title
        actual_positions: (T, 2) actual (lon, lat) degrees
        predicted_positions: dict model_name -> (T, 2) predicted positions
        output_path: save path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))

    for pidx, (mname, pred) in enumerate(predicted_positions.items()):
        n_common = min(len(actual_positions), len(pred))
        # Only compute error from timestep SEQ_LEN onward (first SEQ_LEN are
        # copied from actual).
        if n_common <= SEQ_LEN:
            continue
        timesteps = np.arange(SEQ_LEN, n_common)
        hours = timesteps * HOURS_PER_STEP
        errors = np.array([
            haversine_km(
                actual_positions[t, 0], actual_positions[t, 1],
                pred[t, 0], pred[t, 1],
            )
            for t in timesteps
        ])
        color = PRED_COLORS[pidx % len(PRED_COLORS)]
        ax.plot(hours, errors, color=color, linewidth=1.8, label=mname, marker="o", markersize=3)

    ax.set_xlabel("Hours from storm start", fontsize=10)
    ax.set_ylabel("Position error (km)", fontsize=10)
    ax.set_title(f"Along-track error: {storm_name}", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, framealpha=0.8)
    ax.set_xlim(left=SEQ_LEN * HOURS_PER_STEP)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Internal helpers for generate_test_visualizations
# ---------------------------------------------------------------------------

def _load_storm_data(storm_group: pd.DataFrame, data_root: Path):
    """Load full Data1D track and Env-Data for one storm.

    Returns:
        storm_data dict, env_data_list, or (None, None) on failure.
    """
    storm_group = storm_group.sort_values("timestamp").reset_index(drop=True)
    first = storm_group.iloc[0]

    d1d_path = DATA1D_ROOT / first["basin"] / first["split"] / first["data1d_file"]
    if not d1d_path.exists():
        return None, None

    track_df = pd.read_csv(
        d1d_path, delimiter="\t", header=None, names=DATA1D_COLS,
        dtype={"timestamp": str},
    )
    track_df["timestamp"] = track_df["timestamp"].astype(str).str.strip()

    # Convert to physical units
    features = track_df[DATA1D_FEATURE_COLS].values.astype(np.float32)
    lons = np.array([norm_to_lon(v) for v in track_df["long_norm"]])
    lats = np.array([norm_to_lat(v) for v in track_df["lat_norm"]])
    winds = np.array([norm_to_wind(v) for v in track_df["wnd_norm"]])
    positions = np.column_stack([lons, lats])

    # Load all env-data dicts in track order
    # First, build a mapping timestamp -> env_path from the index
    ts_to_env = {}
    for _, row in storm_group.iterrows():
        ts_to_env[str(row["timestamp"]).strip()] = data_root / row["env_path"]

    env_data_list = []
    for ts in track_df["timestamp"]:
        ts_str = str(ts).strip()
        if ts_str in ts_to_env and ts_to_env[ts_str].exists():
            env_data_list.append(
                np.load(str(ts_to_env[ts_str]), allow_pickle=True).item()
            )
        else:
            # Placeholder with zeros for missing timesteps
            env_data_list.append(None)

    # Pre-load Data3D tensors for timesteps that have them (needed for CNN/Fusion)
    ts_to_d3d = {}
    for _, row in storm_group.iterrows():
        if row.get("data3d_exists", False) and row.get("data3d_path", ""):
            ts_to_d3d[str(row["timestamp"]).strip()] = row["data3d_path"]

    data3d_tensors = {}
    for idx, ts in enumerate(track_df["timestamp"]):
        ts_str = str(ts).strip()
        if ts_str in ts_to_d3d:
            t3d = _load_data3d_tensor(
                data_root, first["basin"], first["year"],
                first["storm_name"], ts_str
            )
            if t3d is not None:
                data3d_tensors[idx] = t3d

    storm_data = {
        "features": features,
        "positions": positions,
        "wind_speeds": winds,
        "timestamps": track_df["timestamp"].values,
        "data3d_tensors": data3d_tensors,
    }
    return storm_data, env_data_list


def _instantiate_model(model_cls_path: str):
    """Dynamically import and instantiate a model class."""
    module_path, cls_name = model_cls_path.rsplit(".", 1)
    import importlib
    mod = importlib.import_module(module_path)
    cls = getattr(mod, cls_name)
    return cls()


def _select_interesting_storms(df: pd.DataFrame, num_storms: int = 6) -> list[tuple]:
    """Select a diverse set of test storms.

    Selection criteria:
    - At least 20 timesteps (5+ days)
    - At least one recurving storm (big latitude change)
    - At least one strong typhoon (high wind)
    - One straight-moving storm for contrast

    Returns list of (year, storm_name) tuples.
    """
    test_df = df[df["split"] == "test"].copy()
    storm_groups = test_df.groupby(["year", "storm_name"])

    candidates = []
    for (year, sname), grp in storm_groups:
        n_ts = len(grp)
        if n_ts < 20:
            continue

        # Load Data1D to compute track properties
        first = grp.iloc[0]
        d1d_path = DATA1D_ROOT / first["basin"] / "test" / first["data1d_file"]
        if not d1d_path.exists():
            continue

        track_df = pd.read_csv(
            d1d_path, delimiter="\t", header=None, names=DATA1D_COLS,
            dtype={"timestamp": str},
        )
        lats = np.array([norm_to_lat(v) for v in track_df["lat_norm"]])
        winds = np.array([norm_to_wind(v) for v in track_df["wnd_norm"]])
        lons = np.array([norm_to_lon(v) for v in track_df["long_norm"]])

        lat_range = lats.max() - lats.min()
        lon_range = lons.max() - lons.min()
        max_wind = winds.max()

        # Recurving heuristic: large latitude change relative to longitude
        recurve_score = lat_range / max(lon_range, 1.0)

        candidates.append({
            "year": year,
            "storm_name": sname,
            "n_timesteps": len(track_df),
            "lat_range": lat_range,
            "lon_range": lon_range,
            "max_wind": max_wind,
            "recurve_score": recurve_score,
        })

    if not candidates:
        # Fall back: take any test storms with at least 10 timesteps
        for (year, sname), grp in storm_groups:
            if len(grp) >= 10:
                candidates.append({
                    "year": year, "storm_name": sname,
                    "n_timesteps": len(grp), "lat_range": 0, "lon_range": 0,
                    "max_wind": 0, "recurve_score": 0,
                })
            if len(candidates) >= num_storms:
                break

    if not candidates:
        return []

    cdf = pd.DataFrame(candidates)
    selected = []

    # 1. Strongest typhoon
    strongest = cdf.loc[cdf["max_wind"].idxmax()]
    selected.append((strongest["year"], strongest["storm_name"]))

    # 2. Most recurving storm
    remaining = cdf[~cdf.apply(lambda r: (r["year"], r["storm_name"]) in selected, axis=1)]
    if len(remaining) > 0:
        recurving = remaining.loc[remaining["recurve_score"].idxmax()]
        selected.append((recurving["year"], recurving["storm_name"]))

    # 3. Straightest-moving storm (lowest recurve score, decent length)
    remaining = cdf[~cdf.apply(lambda r: (r["year"], r["storm_name"]) in selected, axis=1)]
    if len(remaining) > 0:
        straight = remaining.loc[remaining["recurve_score"].idxmin()]
        selected.append((straight["year"], straight["storm_name"]))

    # 4. Fill the rest: longest tracks not yet selected
    remaining = cdf[~cdf.apply(lambda r: (r["year"], r["storm_name"]) in selected, axis=1)]
    remaining = remaining.sort_values("n_timesteps", ascending=False)
    for _, row in remaining.iterrows():
        if len(selected) >= num_storms:
            break
        selected.append((row["year"], row["storm_name"]))

    return selected[:num_storms]


# ---------------------------------------------------------------------------
# 6. Main entry point
# ---------------------------------------------------------------------------

def generate_test_visualizations(
    results_dir: str | Path,
    checkpoint_dir: str | Path,
    index_path: str | Path = MASTER_INDEX_PATH,
    data_root: Path = DATA_ROOT,
    num_storms: int = 6,
):
    """Generate trajectory visualizations for selected test storms.

    1. Loads master index, selects interesting test storms
    2. Loads best checkpoints for each available model
    3. Generates trajectory predictions
    4. Saves individual track plots, multi-storm grid, and error plots

    Args:
        results_dir: root results directory (trajectories/ subfolder created)
        checkpoint_dir: directory containing *_best.pt checkpoint files
        index_path: path to master_index_WP.csv
        data_root: root data directory
        num_storms: how many test storms to visualize
    """
    results_dir = Path(results_dir)
    checkpoint_dir = Path(checkpoint_dir)
    traj_dir = results_dir / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("  TRAJECTORY VISUALIZATION")
    print(f"{'='*60}")

    # Load index
    if not Path(index_path).exists():
        print(f"  Index not found at {index_path}, skipping trajectory visualization.")
        return
    df = pd.read_csv(index_path)
    df = df[df["future_direction24"] >= 0]
    df = df[df["split"].isin(["train", "val", "test"])]

    # Select interesting test storms
    selected = _select_interesting_storms(df, num_storms=num_storms)
    if not selected:
        print("  No suitable test storms found. Skipping trajectory visualization.")
        return
    print(f"  Selected {len(selected)} test storms:")
    for year, sname in selected:
        print(f"    - {sname} ({year})")

    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load available models
    loaded_models = {}
    for model_name, info in MODEL_REGISTRY.items():
        ckpt_path = checkpoint_dir / info["ckpt_pattern"]
        if not ckpt_path.exists():
            continue  # silently skip missing checkpoints
        try:
            model = _instantiate_model(info["model_cls"])
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(device)
            model.eval()
            loaded_models[model_name] = {
                "model": model,
                "model_type": info["model_type"],
                "is_regression": info["is_regression"],
            }
            print(f"  Loaded {model_name} from {ckpt_path.name}")
        except Exception as e:
            print(f"  Failed to load {model_name}: {e}")

    if not loaded_models:
        print("  No models loaded. Skipping trajectory visualization.")
        return

    # Process each storm
    all_storms_viz = []

    for year, storm_name in selected:
        print(f"\n  Processing {storm_name} ({year})...")
        storm_group = df[(df["year"] == year) & (df["storm_name"] == storm_name)]
        if storm_group.empty:
            print(f"    No index entries found, skipping.")
            continue

        storm_data, env_data_list = _load_storm_data(storm_group, data_root)
        if storm_data is None:
            print(f"    Could not load Data1D, skipping.")
            continue

        actual_track = storm_data["positions"]
        wind_speeds = storm_data["wind_speeds"]
        T = len(actual_track)
        print(f"    Track length: {T} timesteps ({T * HOURS_PER_STEP / 24:.1f} days)")

        if T <= SEQ_LEN:
            print(f"    Track too short (need > {SEQ_LEN}), skipping.")
            continue

        # Check if enough env-data is available
        env_available = sum(1 for e in env_data_list if e is not None)

        # Prepare clean env-data (replace None with dummy)
        clean_env = []
        for e in env_data_list:
            clean_env.append(e if e is not None else _make_dummy_env_dict())

        # Predict with each model
        predicted_tracks = {}
        for model_name, minfo in loaded_models.items():
            mtype = minfo["model_type"]
            is_reg = minfo["is_regression"]

            # Check dependencies
            needs_env = mtype in ("env_single", "env_seq", "fusion")
            needs_3d = mtype in ("data3d", "fusion")

            if needs_env and env_available < SEQ_LEN + 1:
                continue
            if needs_3d and not storm_data.get("data3d_tensors"):
                continue

            try:
                if is_reg:
                    pred = predict_trajectory_regression(
                        minfo["model"], storm_data, clean_env, device,
                        model_type=mtype,
                    )
                else:
                    pred = predict_trajectory_classification(
                        minfo["model"], storm_data, clean_env, device,
                        model_type=mtype,
                    )
                predicted_tracks[model_name] = pred
            except Exception as e:
                print(f"    Prediction failed for {model_name}: {e}")

        if not predicted_tracks:
            print(f"    No predictions generated, skipping plots.")
            continue

        # Individual track comparison plot (all models)
        safe_name = f"{storm_name}_{year}".replace(" ", "_")
        plot_storm_track_comparison(
            storm_name=storm_name,
            year=year,
            actual_track=actual_track,
            predicted_tracks=predicted_tracks,
            output_path=traj_dir / f"track_{safe_name}.png",
            wind_speeds=wind_speeds,
        )
        print(f"    Saved track plot: track_{safe_name}.png")

        # Best-2 variation: pick the 2 best REGRESSION models only
        # (classification models use oracle speed, making comparison unfair)
        reg_errors = {}
        for mname, pred in predicted_tracks.items():
            if not mname.startswith("Reg"):
                continue
            n_common = min(len(actual_track), len(pred))
            if n_common > SEQ_LEN:
                dists = [haversine_km(actual_track[i, 0], actual_track[i, 1],
                                      pred[i, 0], pred[i, 1])
                         for i in range(SEQ_LEN, n_common)]
                reg_errors[mname] = np.mean(dists)
        if len(reg_errors) >= 2:
            best_2 = sorted(reg_errors, key=reg_errors.get)[:2]
            best_tracks = {k: predicted_tracks[k] for k in best_2}
            plot_storm_track_comparison(
                storm_name=storm_name,
                year=year,
                actual_track=actual_track,
                predicted_tracks=best_tracks,
                output_path=traj_dir / f"track_best2_{safe_name}.png",
                wind_speeds=wind_speeds,
            )
            print(f"    Saved best-2 plot: track_best2_{safe_name}.png")

        # Error along track
        plot_error_along_track(
            storm_name=f"{storm_name} ({year})",
            actual_positions=actual_track,
            predicted_positions=predicted_tracks,
            output_path=traj_dir / f"error_{safe_name}.png",
        )
        print(f"    Saved error plot: error_{safe_name}.png")

        # Accumulate for grid plot
        all_storms_viz.append({
            "storm_name": storm_name,
            "year": year,
            "actual_track": actual_track,
            "predicted_tracks": predicted_tracks,
            "wind_speeds": wind_speeds,
        })

    # Multi-storm grid (all models)
    if len(all_storms_viz) >= 2:
        ncols = min(3, len(all_storms_viz))
        plot_multi_storm_grid(
            storms_data=all_storms_viz,
            output_path=traj_dir / "multi_storm_grid.png",
            ncols=ncols,
        )
        print(f"\n  Saved multi-storm grid: multi_storm_grid.png")

        # Best-2 grid: for each storm, keep only the 2 best REGRESSION models
        best2_storms = []
        for sv in all_storms_viz:
            actual = sv["actual_track"]
            preds = sv["predicted_tracks"]
            reg_errs = {}
            for mname, pred in preds.items():
                if not mname.startswith("Reg"):
                    continue
                n_common = min(len(actual), len(pred))
                if n_common > SEQ_LEN:
                    dists = [haversine_km(actual[i, 0], actual[i, 1],
                                          pred[i, 0], pred[i, 1])
                             for i in range(SEQ_LEN, n_common)]
                    reg_errs[mname] = np.mean(dists)
            if len(reg_errs) >= 2:
                best_2_names = sorted(reg_errs, key=reg_errs.get)[:2]
                best2_storms.append({
                    **sv,
                    "predicted_tracks": {k: preds[k] for k in best_2_names},
                })
        if len(best2_storms) >= 2:
            plot_multi_storm_grid(
                storms_data=best2_storms,
                output_path=traj_dir / "multi_storm_grid_best2.png",
                ncols=ncols,
            )
            print(f"  Saved best-2 grid: multi_storm_grid_best2.png")

    print(f"\n  Trajectory visualizations saved to: {traj_dir}")
    print(f"{'='*60}\n")


def _make_dummy_env_dict() -> dict:
    """Create a minimal dummy Env-Data dict with zero/default values."""
    return {
        "area": np.zeros(6, dtype=np.float32),
        "wind": 0.0,
        "intensity_class": np.zeros(6, dtype=np.float32),
        "move_velocity": 0.0,
        "month": np.zeros(12, dtype=np.float32),
        "location_long": np.zeros(36, dtype=np.float32),
        "location_lat": np.zeros(12, dtype=np.float32),
        "history_direction12": -1,
        "history_direction24": -1,
        "future_direction24": -1,
        "history_inte_change24": -1,
        "future_inte_change24": -1,
    }
