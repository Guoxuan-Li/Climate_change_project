"""
Comprehensive post-training analysis script.

Produces ALL analysis outputs from existing trained models and raw data,
making the entire project's results fully reproducible from checkpoints.

Outputs:
    1. Track predictions for N test storms (tracks/)
    2. Regression residual analysis (residuals/)
    3. Lon/lat error scatter (scatter/)
    4. Cross-basin decadal intensity trends (cross_basin_intensity/)
    5. Climate change statistical analysis (climate/)
    6. Per-storm summary table (summary/)

Usage:
    python -m src.scripts.run_analysis \\
        --results-dir results/20260325_233354 \\
        --output-dir results/analysis \\
        --num-storms 20
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.integrate import trapezoid
from sklearn.metrics import r2_score

from src.config import (
    BASINS, DATA_ROOT, DATA1D_ROOT, DATA1D_COLS, DATA1D_FEATURE_COLS,
    MASTER_INDEX_PATH, INDEX_DIR, SEQ_LEN, HIDDEN_DIM,
    NORM_LONG, NORM_LAT, NORM_WND, ENV_FEATURE_DIM,
    DATA3D_CHANNELS,
)
from src.training.evaluate import (
    _haversine_km, compute_regression_metrics, _delta_to_direction_class,
)
from src.visualization.trajectory_plots import (
    generate_test_visualizations,
    MODEL_REGISTRY,
    _DarkMapStyle,
    norm_to_lon, norm_to_lat, norm_to_wind,
    haversine_km,
    _select_interesting_storms, _load_storm_data, _make_dummy_env_dict,
    predict_trajectory_regression, predict_trajectory_classification,
    PRED_COLORS, HOURS_PER_STEP,
)
from src.data.utils import env_dict_to_vector

# ---------------------------------------------------------------------------
# Dark-theme style constants (matching intensity_plots.py)
# ---------------------------------------------------------------------------

_BG_COLOR    = "#0E1117"
_AXES_COLOR  = "#1A1D23"
_GRID_COLOR  = "#2A2D35"
_SPINE_COLOR = "#3A3D45"
_FONT_TITLE  = {"fontsize": 14, "fontweight": "bold", "color": "#E0E0E0"}
_FONT_LABEL  = {"fontsize": 11, "color": "#CCCCCC"}
_FONT_TICK   = {"labelsize": 9, "labelcolor": "#AAAAAA"}

NORM_TO_DEG = 5.0           # delta_norm * 5 = delta degrees
WND_SCALE   = 25.0          # wnd_norm = (wnd-40)/25
WND_OFFSET  = 40.0
RI_THRESHOLD_MS = 15.4      # 30 kt / 24 h
CAT4_THRESHOLD_MS = 50.9    # Super Typhoon threshold

# Saffir-Simpson categories
SAFFIR_SIMPSON = [
    ("TD",        0.0,  "#5EBAFF"),
    ("TS",       17.1,  "#00FAF4"),
    ("STS",      24.4,  "#FFFFCC"),
    ("TY",       32.6,  "#FFE775"),
    ("STY",      41.4,  "#FF8F20"),
    ("Super TY", 50.9,  "#FF6060"),
]

# Basin display names
BASIN_NAMES = {
    "EP": "East Pacific",
    "NA": "North Atlantic",
    "NI": "North Indian",
    "SI": "South Indian",
    "SP": "South Pacific",
    "WP": "West Pacific",
}

BASIN_COLORS = {
    "EP": "#FF6B6B",
    "NA": "#4ECDC4",
    "NI": "#FFE66D",
    "SI": "#A8E6CF",
    "SP": "#B8A9C9",
    "WP": "#FF8B94",
}


def _style_ax(ax, xlabel="", ylabel="", title=""):
    """Apply dark styling to a regular axes."""
    ax.set_facecolor(_AXES_COLOR)
    ax.tick_params(**_FONT_TICK, direction="in", length=4, width=0.6,
                   colors="#666666")
    for spine in ax.spines.values():
        spine.set_color(_SPINE_COLOR)
        spine.set_linewidth(0.6)
    ax.grid(True, color=_GRID_COLOR, linewidth=0.4, alpha=0.7)
    if xlabel:
        ax.set_xlabel(xlabel, **_FONT_LABEL)
    if ylabel:
        ax.set_ylabel(ylabel, **_FONT_LABEL)
    if title:
        ax.set_title(title, **_FONT_TITLE, pad=10)


def _saffir_category(wind_ms: float) -> str:
    """Return Saffir-Simpson category name."""
    for name, threshold, _ in reversed(SAFFIR_SIMPSON):
        if wind_ms >= threshold:
            return name
    return "TD"


def _denorm_wind(wnd_norm):
    """Denormalize wind to m/s."""
    return np.asarray(wnd_norm, dtype=float) * WND_SCALE + WND_OFFSET


# ---------------------------------------------------------------------------
# Model loading utilities
# ---------------------------------------------------------------------------

def _instantiate_model(model_cls_path: str):
    """Dynamically import and instantiate a model class."""
    module_path, cls_name = model_cls_path.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    cls = getattr(mod, cls_name)
    return cls()


def _load_checkpoint(model_cls_path: str, ckpt_path: Path, device: torch.device):
    """Instantiate model, load weights, move to device, set eval mode."""
    model = _instantiate_model(model_cls_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Dataset building for batch inference
# ---------------------------------------------------------------------------

def _build_regression_dataloader(index_df: pd.DataFrame, model_type: str,
                                 data_root: Path, batch_size: int = 64):
    """Build a DataLoader for a regression model on the test set.

    Returns (dataloader, dataset) or (None, None) if no data available.
    """
    from torch.utils.data import DataLoader

    test_df = index_df[index_df["split"] == "test"].copy()
    test_df = test_df[
        test_df["delta_lon_norm"].notna() & test_df["delta_lat_norm"].notna()
    ]

    if test_df.empty:
        return None, None

    if model_type == "data1d_seq":
        from src.data.regression_dataset import RegData1DSequenceDataset
        ds = RegData1DSequenceDataset(test_df, data_root=data_root)
    elif model_type == "env_single":
        from src.data.regression_dataset import RegEnvSingleDataset
        ds = RegEnvSingleDataset(test_df, data_root=data_root)
    elif model_type == "env_seq":
        from src.data.regression_dataset import RegEnvSequenceDataset
        ds = RegEnvSequenceDataset(test_df, data_root=data_root)
    elif model_type == "data3d":
        from src.data.regression_dataset import RegData3DDataset
        ds = RegData3DDataset(test_df, data_root=data_root)
    elif model_type == "fusion":
        from src.data.regression_dataset import RegMultimodalDataset
        ds = RegMultimodalDataset(test_df, data_root=data_root)
    else:
        return None, None

    if len(ds) == 0:
        return None, None

    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return dl, ds


@torch.no_grad()
def _collect_predictions(model, dataloader, device: torch.device):
    """Run a regression model on a dataloader and collect (y_true, y_pred).

    Returns (y_true, y_pred) both as (N, 2) numpy arrays.
    """
    model.eval()
    all_preds, all_targets = [], []

    for batch in dataloader:
        if len(batch) == 2:
            features, targets = batch
        else:
            *features_list, targets = batch
            features = features_list

        if isinstance(features, (list, tuple)):
            features = [f.to(device) for f in features]
            preds = model(*features)
        else:
            features = features.to(device)
            preds = model(features)

        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    return y_true, y_pred


# ===================================================================
# Section 1: Track Predictions
# ===================================================================

def run_track_predictions(results_dir: Path, checkpoint_dir: Path,
                          output_dir: Path, num_storms: int = 20):
    """Generate track prediction visualizations for test storms.

    Uses the existing generate_test_visualizations() with the requested
    number of storms. Outputs go to {output_dir}/tracks/.
    """
    print("\n" + "=" * 70)
    print("  SECTION 1: Track Predictions")
    print("=" * 70)

    tracks_dir = output_dir / "tracks"
    tracks_dir.mkdir(parents=True, exist_ok=True)

    # generate_test_visualizations writes to results_dir/trajectories/
    # We create a temporary wrapper results_dir pointing at tracks_dir parent
    # so that trajectories/ subfolder lands inside our output.
    # Actually, we can just pass results_dir=tracks_dir and it will create
    # tracks_dir/trajectories/.  Flatten by using the tracks_dir directly.
    generate_test_visualizations(
        results_dir=tracks_dir,
        checkpoint_dir=checkpoint_dir,
        index_path=MASTER_INDEX_PATH,
        data_root=DATA_ROOT,
        num_storms=num_storms,
    )
    print(f"  Track predictions saved to: {tracks_dir}")


# ===================================================================
# Section 2: Regression Residual Analysis
# ===================================================================

def run_residual_analysis(checkpoint_dir: Path, output_dir: Path,
                          index_df: pd.DataFrame, device: torch.device):
    """Compute and visualize residuals for Reg CNN and Reg LSTM.

    Produces:
        - residual_histograms.png  (2x2: CNN dlon, CNN dlat, LSTM dlon, LSTM dlat)
        - residual_vs_actual.png   (2x2 scatter for heteroscedasticity check)
    """
    print("\n" + "=" * 70)
    print("  SECTION 2: Regression Residual Analysis")
    print("=" * 70)

    resid_dir = output_dir / "residuals"
    resid_dir.mkdir(parents=True, exist_ok=True)

    # Models to analyze: CNN (stage4) and LSTM (stage2)
    models_to_load = {
        "Reg CNN": {
            "ckpt": "reg_stage4_cnn_3d_best.pt",
            "cls": "src.models.regression_models.RegCNN3D",
            "type": "data3d",
        },
        "Reg LSTM": {
            "ckpt": "reg_stage2_lstm_1d_best.pt",
            "cls": "src.models.regression_models.RegLSTM",
            "type": "data1d_seq",
        },
    }

    results = {}  # model_name -> {y_true, y_pred, residuals}

    for mname, info in models_to_load.items():
        ckpt_path = checkpoint_dir / info["ckpt"]
        if not ckpt_path.exists():
            print(f"  Checkpoint not found: {ckpt_path}, skipping {mname}")
            continue

        print(f"  Loading {mname}...")
        model = _load_checkpoint(info["cls"], ckpt_path, device)

        dl, ds = _build_regression_dataloader(
            index_df, info["type"], DATA_ROOT, batch_size=32 if "CNN" in mname else 64
        )
        if dl is None:
            print(f"  No test data for {mname}, skipping.")
            continue

        print(f"  Running inference ({len(ds)} samples)...")
        y_true, y_pred = _collect_predictions(model, dl, device)
        residuals = y_pred - y_true  # (N, 2): [dlon_resid, dlat_resid]

        results[mname] = {
            "y_true": y_true,
            "y_pred": y_pred,
            "residuals": residuals,
        }
        print(f"  {mname}: mean residual dlon={residuals[:, 0].mean():.4f}, "
              f"dlat={residuals[:, 1].mean():.4f}")

        # Free GPU memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    if not results:
        print("  No models loaded, skipping residual analysis.")
        return results

    # --- Plot 1: Residual Histograms (2x2) ---
    model_names = list(results.keys())
    n_models = len(model_names)

    with plt.style.context("dark_background"):
        fig, axes = plt.subplots(n_models, 2, figsize=(12, 4.5 * n_models),
                                 facecolor=_BG_COLOR)
        if n_models == 1:
            axes = axes[np.newaxis, :]

        for row, mname in enumerate(model_names):
            resid = results[mname]["residuals"]
            for col, (comp, label) in enumerate([(0, "delta_lon_norm"),
                                                  (1, "delta_lat_norm")]):
                ax = axes[row, col]
                _style_ax(ax, xlabel="Residual (pred - actual)",
                          ylabel="Frequency")
                r = resid[:, comp]
                mu, sigma = r.mean(), r.std()

                ax.hist(r, bins=80, color=PRED_COLORS[row % len(PRED_COLORS)],
                        alpha=0.75, edgecolor="white", linewidth=0.3,
                        density=True, zorder=3)

                # Overlay a normal PDF for reference
                x_range = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
                pdf = stats.norm.pdf(x_range, mu, sigma)
                ax.plot(x_range, pdf, color="white", linewidth=1.5,
                        linestyle="--", alpha=0.8, zorder=4)

                ax.axvline(0, color="#FFE66D", linewidth=1.0, linestyle=":",
                           alpha=0.7, zorder=4)

                # Annotation
                txt = (f"mean = {mu:.4f}\n"
                       f"std  = {sigma:.4f}\n"
                       f"N    = {len(r)}")
                ax.text(0.97, 0.95, txt, transform=ax.transAxes,
                        fontsize=9, color="#CCCCCC", va="top", ha="right",
                        family="monospace",
                        bbox=dict(facecolor=_AXES_COLOR, alpha=0.8,
                                  edgecolor=_SPINE_COLOR, pad=5))

                ax.set_title(f"{mname} — {label}", **_FONT_TITLE, pad=8)

        fig.suptitle("Regression Residuals (pred - actual)",
                     fontsize=16, fontweight="bold", color="#E0E0E0", y=1.01)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        fig.savefig(resid_dir / "residual_histograms.png", dpi=200,
                    bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
    print(f"  Saved: residual_histograms.png")

    # --- Plot 2: Residual vs Actual (heteroscedasticity check) ---
    with plt.style.context("dark_background"):
        fig, axes = plt.subplots(n_models, 2, figsize=(12, 4.5 * n_models),
                                 facecolor=_BG_COLOR)
        if n_models == 1:
            axes = axes[np.newaxis, :]

        for row, mname in enumerate(model_names):
            r = results[mname]
            for col, (comp, label) in enumerate([(0, "delta_lon_norm"),
                                                  (1, "delta_lat_norm")]):
                ax = axes[row, col]
                _style_ax(ax, xlabel=f"Actual {label}",
                          ylabel="Residual (pred - actual)")

                actual = r["y_true"][:, comp]
                resid = r["residuals"][:, comp]

                ax.scatter(actual, resid, s=3, alpha=0.25,
                           color=PRED_COLORS[row % len(PRED_COLORS)],
                           zorder=3, rasterized=True)

                # Zero line
                ax.axhline(0, color="#FFE66D", linewidth=1.0, linestyle="--",
                           alpha=0.7, zorder=4)

                # LOESS-like trend: bin actual values and compute mean residual
                n_bins = 30
                bin_edges = np.percentile(actual, np.linspace(0, 100, n_bins + 1))
                bin_centers, bin_means, bin_stds = [], [], []
                for i in range(n_bins):
                    mask = (actual >= bin_edges[i]) & (actual < bin_edges[i + 1])
                    if mask.sum() > 5:
                        bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
                        bin_means.append(resid[mask].mean())
                        bin_stds.append(resid[mask].std())
                if bin_centers:
                    ax.plot(bin_centers, bin_means, color="white", linewidth=2.0,
                            zorder=5, label="Binned mean")
                    ax.fill_between(bin_centers,
                                    np.array(bin_means) - np.array(bin_stds),
                                    np.array(bin_means) + np.array(bin_stds),
                                    color="white", alpha=0.1, zorder=2)

                ax.set_title(f"{mname} — {label}", **_FONT_TITLE, pad=8)
                ax.legend(fontsize=8, framealpha=0.7, facecolor=_AXES_COLOR,
                          edgecolor=_SPINE_COLOR, labelcolor="white",
                          loc="upper left")

        fig.suptitle("Residual vs Actual (Heteroscedasticity Check)",
                     fontsize=16, fontweight="bold", color="#E0E0E0", y=1.01)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        fig.savefig(resid_dir / "residual_vs_actual.png", dpi=200,
                    bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
    print(f"  Saved: residual_vs_actual.png")

    return results


# ===================================================================
# Section 3: Lon vs Lat Error Scatter
# ===================================================================

def run_scatter_analysis(residual_results: dict, output_dir: Path):
    """Scatter: actual vs predicted for delta_lon and delta_lat.

    Produces a 2x2 figure (CNN lon, CNN lat, LSTM lon, LSTM lat) with
    y=x reference line, R2, and MAE annotations.
    """
    print("\n" + "=" * 70)
    print("  SECTION 3: Lon vs Lat Error Scatter")
    print("=" * 70)

    scatter_dir = output_dir / "scatter"
    scatter_dir.mkdir(parents=True, exist_ok=True)

    model_names = list(residual_results.keys())
    if not model_names:
        print("  No residual data available, skipping scatter analysis.")
        return

    n_models = len(model_names)
    with plt.style.context("dark_background"):
        fig, axes = plt.subplots(n_models, 2, figsize=(12, 5.5 * n_models),
                                 facecolor=_BG_COLOR)
        if n_models == 1:
            axes = axes[np.newaxis, :]

        for row, mname in enumerate(model_names):
            r = residual_results[mname]
            y_true = r["y_true"]
            y_pred = r["y_pred"]

            for col, (comp, label) in enumerate([(0, "delta_lon_norm"),
                                                  (1, "delta_lat_norm")]):
                ax = axes[row, col]
                _style_ax(ax, xlabel=f"Actual {label}",
                          ylabel=f"Predicted {label}")

                actual = y_true[:, comp]
                predicted = y_pred[:, comp]

                # Compute metrics
                r2 = r2_score(actual, predicted)
                mae = np.mean(np.abs(predicted - actual))

                # Scatter with density-based alpha
                ax.scatter(actual, predicted, s=3, alpha=0.2,
                           color=PRED_COLORS[row % len(PRED_COLORS)],
                           zorder=3, rasterized=True)

                # y=x reference line
                lims = [min(actual.min(), predicted.min()) - 0.2,
                        max(actual.max(), predicted.max()) + 0.2]
                ax.plot(lims, lims, color="#FFE66D", linewidth=1.5,
                        linestyle="--", alpha=0.8, zorder=4, label="y = x")

                # Annotation
                txt = f"R$^2$ = {r2:.4f}\nMAE = {mae:.4f}"
                ax.text(0.05, 0.95, txt, transform=ax.transAxes,
                        fontsize=10, color="#CCCCCC", va="top", ha="left",
                        family="monospace",
                        bbox=dict(facecolor=_AXES_COLOR, alpha=0.8,
                                  edgecolor=_SPINE_COLOR, pad=5))

                ax.set_xlim(lims)
                ax.set_ylim(lims)
                ax.set_aspect("equal", adjustable="box")
                ax.set_title(f"{mname} — {label}", **_FONT_TITLE, pad=8)
                ax.legend(fontsize=8, framealpha=0.7, facecolor=_AXES_COLOR,
                          edgecolor=_SPINE_COLOR, labelcolor="white",
                          loc="lower right")

        fig.suptitle("Actual vs Predicted Displacement",
                     fontsize=16, fontweight="bold", color="#E0E0E0", y=1.01)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        fig.savefig(scatter_dir / "actual_vs_predicted_scatter.png", dpi=200,
                    bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
    print(f"  Saved: actual_vs_predicted_scatter.png")


# ===================================================================
# Section 4: Cross-Basin Decadal Intensity Trends
# ===================================================================

def _load_basin_storm_stats(basin: str) -> pd.DataFrame:
    """Load Data1D files for a basin and compute per-storm intensity metrics.

    Returns a DataFrame with columns:
        basin, year, storm_name, peak_wind, duration_h, integrated_intensity,
        ace, ri_events, is_cat4plus, max_24h_delta, decade
    """
    index_path = INDEX_DIR / f"master_index_{basin}.csv"
    if not index_path.exists():
        return pd.DataFrame()

    try:
        idx_df = pd.read_csv(index_path)
    except Exception:
        return pd.DataFrame()

    # Get unique storms
    storms = idx_df.groupby(["year", "storm_name"]).first().reset_index()
    records = []

    for _, row in storms.iterrows():
        split_dir = row.get("split", "train")
        if split_dir == "unknown" or pd.isna(split_dir):
            split_dir = "train"
        d1d_file = row.get("data1d_file", "")
        if pd.isna(d1d_file) or not d1d_file:
            continue

        d1d_path = DATA1D_ROOT / basin / split_dir / d1d_file
        if not d1d_path.exists():
            # Try other splits
            for alt_split in ["train", "val", "test"]:
                alt_path = DATA1D_ROOT / basin / alt_split / d1d_file
                if alt_path.exists():
                    d1d_path = alt_path
                    break
            else:
                continue

        try:
            track_df = pd.read_csv(
                d1d_path, delimiter="\t", header=None, names=DATA1D_COLS,
                dtype={"timestamp": str},
            )
        except Exception:
            continue

        winds_ms = _denorm_wind(track_df["wnd_norm"].values)
        n_ts = len(winds_ms)
        if n_ts < 2:
            continue

        peak_wind = float(winds_ms.max())
        duration_h = (n_ts - 1) * 6  # hours

        # Integrated intensity: trapezoid integration of wind over time
        time_hours = np.arange(n_ts) * 6.0
        integrated = float(trapezoid(winds_ms, time_hours))

        # ACE: sum of v^2 for all 6-hourly periods where v >= 17.5 m/s (TS+)
        # ACE is typically in 10^4 kt^2, but we compute in m/s^2 * count
        ace = float(np.sum(winds_ms[winds_ms >= 17.5] ** 2))

        # 24h deltas: wind[t+4] - wind[t]
        deltas_24h = []
        for t in range(n_ts - 4):
            deltas_24h.append(winds_ms[t + 4] - winds_ms[t])
        deltas_24h = np.array(deltas_24h) if deltas_24h else np.array([0.0])

        ri_events = int(np.sum(deltas_24h >= RI_THRESHOLD_MS))
        max_24h_delta = float(deltas_24h.max()) if len(deltas_24h) > 0 else 0.0

        year = int(row["year"])
        decade = (year // 10) * 10

        records.append({
            "basin": basin,
            "year": year,
            "storm_name": row["storm_name"],
            "peak_wind": peak_wind,
            "duration_h": duration_h,
            "integrated_intensity": integrated,
            "ace": ace,
            "ri_events": ri_events,
            "is_cat4plus": peak_wind >= CAT4_THRESHOLD_MS,
            "max_24h_delta": max_24h_delta,
            "decade": decade,
            "n_timesteps": n_ts,
        })

    return pd.DataFrame(records)


def run_cross_basin_intensity(output_dir: Path):
    """Compute and plot cross-basin decadal intensity trends.

    Produces:
        - cat4_fraction_by_basin.png   (6 lines)
        - ri_prevalence_by_basin.png   (6 lines)
        - per_basin_decadal_bars.png   (2x3 grid)
    """
    print("\n" + "=" * 70)
    print("  SECTION 4: Cross-Basin Decadal Intensity Trends")
    print("=" * 70)

    cb_dir = output_dir / "cross_basin_intensity"
    cb_dir.mkdir(parents=True, exist_ok=True)

    # Load all basins
    all_stats = {}
    for basin in BASINS:
        print(f"  Loading {basin}...")
        df = _load_basin_storm_stats(basin)
        if not df.empty:
            all_stats[basin] = df
            print(f"    {len(df)} storms loaded")
        else:
            print(f"    No data found, skipping")

    if not all_stats:
        print("  No basin data available, skipping cross-basin analysis.")
        return

    # Combine all for CSV export
    combined = pd.concat(all_stats.values(), ignore_index=True)
    combined.to_csv(cb_dir / "all_basin_storm_stats.csv", index=False)
    print(f"  Saved: all_basin_storm_stats.csv ({len(combined)} storms)")

    # --- Plot 1: Cat4+ fraction trend by basin ---
    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(12, 6), facecolor=_BG_COLOR)
        _style_ax(ax, xlabel="Decade", ylabel="Cat 4+ Fraction (%)",
                  title="Category 4+ Storm Fraction by Basin and Decade")

        for basin, df in all_stats.items():
            decade_groups = df.groupby("decade")
            decades = sorted(df["decade"].unique())
            fracs = []
            valid_decades = []
            for d in decades:
                g = decade_groups.get_group(d)
                if len(g) >= 3:  # minimum storms per decade
                    fracs.append(g["is_cat4plus"].mean() * 100)
                    valid_decades.append(d)

            if valid_decades:
                ax.plot(valid_decades, fracs,
                        color=BASIN_COLORS.get(basin, "#FFFFFF"),
                        linewidth=2.0, marker="o", markersize=5,
                        label=f"{basin} ({BASIN_NAMES.get(basin, basin)})",
                        zorder=3)

        ax.legend(fontsize=9, framealpha=0.7, facecolor=_AXES_COLOR,
                  edgecolor=_SPINE_COLOR, labelcolor="white",
                  loc="upper left")
        plt.tight_layout()
        fig.savefig(cb_dir / "cat4_fraction_by_basin.png", dpi=200,
                    bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
    print(f"  Saved: cat4_fraction_by_basin.png")

    # --- Plot 2: RI prevalence by basin ---
    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(12, 6), facecolor=_BG_COLOR)
        _style_ax(ax, xlabel="Decade", ylabel="RI Events per Storm",
                  title="Rapid Intensification Prevalence by Basin and Decade")

        for basin, df in all_stats.items():
            decade_groups = df.groupby("decade")
            decades = sorted(df["decade"].unique())
            ri_rates = []
            valid_decades = []
            for d in decades:
                g = decade_groups.get_group(d)
                if len(g) >= 3:
                    ri_rates.append(g["ri_events"].sum() / len(g))
                    valid_decades.append(d)

            if valid_decades:
                ax.plot(valid_decades, ri_rates,
                        color=BASIN_COLORS.get(basin, "#FFFFFF"),
                        linewidth=2.0, marker="s", markersize=5,
                        label=f"{basin} ({BASIN_NAMES.get(basin, basin)})",
                        zorder=3)

        ax.legend(fontsize=9, framealpha=0.7, facecolor=_AXES_COLOR,
                  edgecolor=_SPINE_COLOR, labelcolor="white",
                  loc="upper left")
        plt.tight_layout()
        fig.savefig(cb_dir / "ri_prevalence_by_basin.png", dpi=200,
                    bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
    print(f"  Saved: ri_prevalence_by_basin.png")

    # --- Plot 3: Per-basin decadal bar charts (3x2 grid) ---
    with plt.style.context("dark_background"):
        n_basins = len(all_stats)
        ncols = 3
        nrows = math.ceil(n_basins / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows),
                                 facecolor=_BG_COLOR)
        if nrows == 1:
            axes = axes[np.newaxis, :] if ncols > 1 else np.array([[axes]])
        elif ncols == 1:
            axes = axes[:, np.newaxis]

        for idx, (basin, df) in enumerate(sorted(all_stats.items())):
            r, c = divmod(idx, ncols)
            ax = axes[r, c]
            _style_ax(ax, ylabel="Mean Peak Wind (m/s)")

            decade_groups = df.groupby("decade")
            decades = sorted(df["decade"].unique())
            labels, values, counts = [], [], []
            for d in decades:
                g = decade_groups.get_group(d)
                if len(g) >= 3:
                    labels.append(f"{d}s")
                    values.append(g["peak_wind"].mean())
                    counts.append(len(g))

            if labels:
                bars = ax.bar(labels, values,
                              color=BASIN_COLORS.get(basin, "#FFFFFF"),
                              alpha=0.85, edgecolor="white", linewidth=0.4,
                              zorder=3)
                # Storm count labels
                for i, (v, n) in enumerate(zip(values, counts)):
                    ax.text(i, v + 0.5, f"n={n}", ha="center",
                            fontsize=7, color="#AAAAAA")

            ax.set_title(f"{basin} — {BASIN_NAMES.get(basin, basin)}",
                         **_FONT_TITLE, pad=8)
            ax.tick_params(axis="x", labelrotation=45)

        # Hide unused axes
        for idx in range(len(all_stats), nrows * ncols):
            r, c = divmod(idx, ncols)
            axes[r, c].set_visible(False)

        fig.suptitle("Per-Basin Mean Peak Wind by Decade",
                     fontsize=16, fontweight="bold", color="#E0E0E0", y=1.01)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(cb_dir / "per_basin_decadal_bars.png", dpi=200,
                    bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
    print(f"  Saved: per_basin_decadal_bars.png")


# ===================================================================
# Section 5: Climate Change Statistical Analysis
# ===================================================================

def run_climate_analysis(output_dir: Path, basin: str = "WP"):
    """Comprehensive climate trend analysis for a basin.

    Produces:
        - intensity_climate_trends.png      (6-panel)
        - intensity_satellite_era_comparison.png (4-panel three-period comparison)
        - cat4_and_ri_trends.png            (2-panel)
        - climate_statistics.csv            (p-values, slopes)
        - storm_stats.csv                   (per-storm raw data)
        - decade_stats.csv                  (per-decade aggregated)
    """
    print("\n" + "=" * 70)
    print("  SECTION 5: Climate Change Statistical Analysis")
    print("=" * 70)

    clim_dir = output_dir / "climate"
    clim_dir.mkdir(parents=True, exist_ok=True)

    # Load per-storm data
    storm_df = _load_basin_storm_stats(basin)
    if storm_df.empty:
        print(f"  No storm data for {basin}, skipping climate analysis.")
        return

    storm_df.to_csv(clim_dir / "storm_stats.csv", index=False)
    print(f"  Loaded {len(storm_df)} storms from {basin}")

    # --- Per-storm additional metrics ---
    # RI flag: has at least one RI event
    storm_df["has_ri"] = storm_df["ri_events"] > 0

    # Compute per-decade aggregations
    decade_groups = storm_df.groupby("decade")
    decade_records = []
    for decade, grp in decade_groups:
        if len(grp) < 3:
            continue
        decade_records.append({
            "decade": decade,
            "n_storms": len(grp),
            "mean_peak_wind": grp["peak_wind"].mean(),
            "median_peak_wind": grp["peak_wind"].median(),
            "mean_integrated": grp["integrated_intensity"].mean(),
            "median_integrated": grp["integrated_intensity"].median(),
            "cat4_fraction": grp["is_cat4plus"].mean() * 100,
            "ri_fraction": grp["has_ri"].mean() * 100,
            "total_ri_events": grp["ri_events"].sum(),
            "ri_per_storm": grp["ri_events"].sum() / len(grp),
            "mean_max_24h_delta": grp["max_24h_delta"].mean(),
            "mean_ace": grp["ace"].mean(),
        })

    decade_df = pd.DataFrame(decade_records)
    decade_df.to_csv(clim_dir / "decade_stats.csv", index=False)
    print(f"  Computed {len(decade_df)} decades of statistics")

    # --- Statistical tests ---
    stat_results = {}

    # Test 1: Linear trend in peak wind over years
    years = storm_df["year"].values.astype(float)
    peak_winds = storm_df["peak_wind"].values

    slope, intercept, r_value, p_value, std_err = stats.linregress(years, peak_winds)
    stat_results["peak_wind_trend"] = {
        "slope": slope, "intercept": intercept,
        "r_value": r_value, "p_value": p_value, "std_err": std_err,
        "description": "Linear regression of peak wind vs year",
    }

    # Test 2: Linear trend in integrated intensity
    integrated = storm_df["integrated_intensity"].values
    slope_i, intercept_i, r_i, p_i, se_i = stats.linregress(years, integrated)
    stat_results["integrated_intensity_trend"] = {
        "slope": slope_i, "intercept": intercept_i,
        "r_value": r_i, "p_value": p_i, "std_err": se_i,
        "description": "Linear regression of integrated intensity vs year",
    }

    # Test 3: Mann-Whitney U test comparing early (pre-1990) vs late (post-2000)
    early = storm_df[storm_df["year"] < 1990]["peak_wind"].values
    late = storm_df[storm_df["year"] >= 2000]["peak_wind"].values
    if len(early) > 5 and len(late) > 5:
        mw_stat, mw_p = stats.mannwhitneyu(early, late, alternative="two-sided")
        stat_results["mannwhitney_peak_wind_early_vs_late"] = {
            "statistic": float(mw_stat), "p_value": float(mw_p),
            "early_median": float(np.median(early)),
            "late_median": float(np.median(late)),
            "description": "Mann-Whitney U: peak wind pre-1990 vs post-2000",
        }

    # Test 4: Cat4+ proportion trend (logistic / chi-squared)
    if len(decade_df) >= 3:
        d_years = decade_df["decade"].values.astype(float)
        d_cat4 = decade_df["cat4_fraction"].values
        slope_c, intercept_c, r_c, p_c, se_c = stats.linregress(d_years, d_cat4)
        stat_results["cat4_trend"] = {
            "slope_per_decade": slope_c * 10,
            "p_value": p_c, "r_value": r_c,
            "description": "Linear regression of Cat4+ % vs decade",
        }

    # Test 5: RI prevalence trend
    if len(decade_df) >= 3:
        d_ri = decade_df["ri_fraction"].values
        slope_r, intercept_r, r_r, p_r, se_r = stats.linregress(d_years, d_ri)
        stat_results["ri_trend"] = {
            "slope_per_decade": slope_r * 10,
            "p_value": p_r, "r_value": r_r,
            "description": "Linear regression of RI% vs decade",
        }

    # Test 6: Three-period comparison (satellite era)
    periods = {
        "1980-1995": storm_df[(storm_df["year"] >= 1980) & (storm_df["year"] <= 1995)],
        "1996-2010": storm_df[(storm_df["year"] >= 1996) & (storm_df["year"] <= 2010)],
        "2011-2023": storm_df[(storm_df["year"] >= 2011) & (storm_df["year"] <= 2023)],
    }
    for metric in ["peak_wind", "integrated_intensity"]:
        period_data = {k: v[metric].values for k, v in periods.items() if len(v) > 0}
        groups = list(period_data.values())
        if len(groups) >= 2:
            try:
                h_stat, h_p = stats.kruskal(*groups)
                stat_results[f"kruskal_{metric}_3periods"] = {
                    "statistic": float(h_stat), "p_value": float(h_p),
                    "period_medians": {k: float(np.median(v))
                                       for k, v in period_data.items()},
                    "description": f"Kruskal-Wallis: {metric} across 3 satellite-era periods",
                }
            except Exception:
                pass

    # Save statistics
    # Convert numpy types for JSON serialization
    def _to_json_safe(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    safe_stats = {}
    for k, v in stat_results.items():
        safe_stats[k] = {k2: _to_json_safe(v2) for k2, v2 in v.items()}

    with open(clim_dir / "climate_statistics.json", "w") as f:
        json.dump(safe_stats, f, indent=2, default=str)

    # Also save as CSV for easy viewing
    csv_rows = []
    for test_name, vals in stat_results.items():
        row = {"test": test_name}
        for k2, v2 in vals.items():
            if isinstance(v2, dict):
                for k3, v3 in v2.items():
                    row[f"{k2}_{k3}"] = v3
            else:
                row[k2] = v2
        csv_rows.append(row)
    pd.DataFrame(csv_rows).to_csv(clim_dir / "climate_statistics.csv", index=False)
    print(f"  Saved: climate_statistics.csv / .json")

    # === PLOTS ===

    # --- Plot 1: intensity_climate_trends.png (6-panel) ---
    with plt.style.context("dark_background"):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor=_BG_COLOR)

        # Panel 1: Mean integrated intensity by decade
        ax = axes[0, 0]
        _style_ax(ax, xlabel="Decade", ylabel="Mean Integrated Intensity (m*s*h)")
        if len(decade_df) > 0:
            ax.bar([f"{d}s" for d in decade_df["decade"]],
                   decade_df["mean_integrated"],
                   color="#4ECDC4", alpha=0.85, edgecolor="white",
                   linewidth=0.4, zorder=3)
        ax.set_title("Mean Integrated Intensity", **_FONT_TITLE, pad=8)
        ax.tick_params(axis="x", labelrotation=45)

        # Panel 2: Median integrated intensity by decade
        ax = axes[0, 1]
        _style_ax(ax, xlabel="Decade", ylabel="Median Integrated Intensity")
        if len(decade_df) > 0:
            ax.bar([f"{d}s" for d in decade_df["decade"]],
                   decade_df["median_integrated"],
                   color="#FF6B6B", alpha=0.85, edgecolor="white",
                   linewidth=0.4, zorder=3)
        ax.set_title("Median Integrated Intensity", **_FONT_TITLE, pad=8)
        ax.tick_params(axis="x", labelrotation=45)

        # Panel 3: Cat4+ fraction
        ax = axes[0, 2]
        _style_ax(ax, xlabel="Decade", ylabel="Cat 4+ Fraction (%)")
        if len(decade_df) > 0:
            ax.bar([f"{d}s" for d in decade_df["decade"]],
                   decade_df["cat4_fraction"],
                   color="#FF8F20", alpha=0.85, edgecolor="white",
                   linewidth=0.4, zorder=3)
        ax.set_title("Cat 4+ Fraction", **_FONT_TITLE, pad=8)
        ax.tick_params(axis="x", labelrotation=45)

        # Panel 4: Peak wind scatter + trend line
        ax = axes[1, 0]
        _style_ax(ax, xlabel="Year", ylabel="Peak Wind (m/s)")
        ax.scatter(storm_df["year"], storm_df["peak_wind"],
                   s=8, alpha=0.3, color="#4ECDC4", zorder=3, rasterized=True)
        # Trend line
        if "peak_wind_trend" in stat_results:
            s = stat_results["peak_wind_trend"]
            x_line = np.array([years.min(), years.max()])
            y_line = s["slope"] * x_line + s["intercept"]
            ax.plot(x_line, y_line, color="#FFE66D", linewidth=2.0,
                    linestyle="--", zorder=4,
                    label=f"slope={s['slope']:.3f}, p={s['p_value']:.3f}")
            ax.legend(fontsize=8, framealpha=0.7, facecolor=_AXES_COLOR,
                      edgecolor=_SPINE_COLOR, labelcolor="white")
        ax.set_title("Peak Wind vs Year (Scatter + Trend)", **_FONT_TITLE, pad=8)

        # Panel 5: Mean peak 24h delta
        ax = axes[1, 1]
        _style_ax(ax, xlabel="Decade", ylabel="Mean Max 24h Delta (m/s)")
        if len(decade_df) > 0:
            colors_bar = ["#E74C3C" if v >= 0 else "#3498DB"
                          for v in decade_df["mean_max_24h_delta"]]
            ax.bar([f"{d}s" for d in decade_df["decade"]],
                   decade_df["mean_max_24h_delta"],
                   color=colors_bar, alpha=0.85, edgecolor="white",
                   linewidth=0.4, zorder=3)
        ax.set_title("Mean Max 24h Intensification", **_FONT_TITLE, pad=8)
        ax.tick_params(axis="x", labelrotation=45)

        # Panel 6: RI fraction
        ax = axes[1, 2]
        _style_ax(ax, xlabel="Decade", ylabel="RI Storms (%)")
        if len(decade_df) > 0:
            ax.bar([f"{d}s" for d in decade_df["decade"]],
                   decade_df["ri_fraction"],
                   color="#FF6060", alpha=0.85, edgecolor="white",
                   linewidth=0.4, zorder=3)
        ax.set_title("Rapid Intensification Prevalence", **_FONT_TITLE, pad=8)
        ax.tick_params(axis="x", labelrotation=45)

        fig.suptitle(f"{BASIN_NAMES.get(basin, basin)} — Intensity Climate Trends",
                     fontsize=16, fontweight="bold", color="#E0E0E0", y=1.01)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(clim_dir / "intensity_climate_trends.png", dpi=200,
                    bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
    print(f"  Saved: intensity_climate_trends.png")

    # --- Plot 2: intensity_satellite_era_comparison.png (4-panel) ---
    with plt.style.context("dark_background"):
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), facecolor=_BG_COLOR)

        period_names = ["1980-1995", "1996-2010", "2011-2023"]
        period_colors = ["#4ECDC4", "#FFE66D", "#FF6B6B"]
        period_dfs = [periods.get(p, pd.DataFrame()) for p in period_names]

        metrics_to_compare = [
            ("peak_wind", "Peak Wind (m/s)", axes[0, 0]),
            ("integrated_intensity", "Integrated Intensity", axes[0, 1]),
            ("max_24h_delta", "Max 24h Intensification (m/s)", axes[1, 0]),
            ("duration_h", "Storm Duration (hours)", axes[1, 1]),
        ]

        for metric, ylabel, ax in metrics_to_compare:
            _style_ax(ax, ylabel=ylabel)

            bp_data = []
            bp_labels = []
            for pname, pdf in zip(period_names, period_dfs):
                if not pdf.empty and metric in pdf.columns:
                    bp_data.append(pdf[metric].dropna().values)
                    bp_labels.append(f"{pname}\n(n={len(pdf)})")
                else:
                    bp_data.append(np.array([]))
                    bp_labels.append(f"{pname}\n(n=0)")

            # Box plot
            if any(len(d) > 0 for d in bp_data):
                bp = ax.boxplot(
                    [d for d in bp_data if len(d) > 0],
                    tick_labels=[l for d, l in zip(bp_data, bp_labels) if len(d) > 0],
                    patch_artist=True, widths=0.5,
                    medianprops=dict(color="white", linewidth=2),
                    whiskerprops=dict(color="#AAAAAA"),
                    capprops=dict(color="#AAAAAA"),
                    flierprops=dict(marker="o", markerfacecolor="#666666",
                                    markersize=3, alpha=0.5),
                )
                valid_colors = [c for d, c in zip(bp_data, period_colors) if len(d) > 0]
                for patch, color in zip(bp["boxes"], valid_colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.6)
                    patch.set_edgecolor("white")

            ax.set_title(metric.replace("_", " ").title(), **_FONT_TITLE, pad=8)

        fig.suptitle(f"{BASIN_NAMES.get(basin, basin)} — Satellite Era Comparison",
                     fontsize=16, fontweight="bold", color="#E0E0E0", y=1.01)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(clim_dir / "intensity_satellite_era_comparison.png", dpi=200,
                    bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
    print(f"  Saved: intensity_satellite_era_comparison.png")

    # --- Plot 3: cat4_and_ri_trends.png (2-panel) ---
    with plt.style.context("dark_background"):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), facecolor=_BG_COLOR)

        if len(decade_df) >= 3:
            d_years = decade_df["decade"].values.astype(float)

            # Cat4+ panel
            ax = axes[0]
            _style_ax(ax, xlabel="Decade", ylabel="Cat 4+ Fraction (%)")
            d_labels = [f"{d}s" for d in decade_df["decade"]]
            ax.bar(d_labels, decade_df["cat4_fraction"],
                   color="#FF8F20", alpha=0.85, edgecolor="white",
                   linewidth=0.4, zorder=3)
            # Trend line
            if "cat4_trend" in stat_results:
                lr = stats.linregress(d_years, decade_df["cat4_fraction"].values)
                x_pos = np.arange(len(d_labels))
                y_fit = lr.slope * d_years + lr.intercept
                ax.plot(x_pos, y_fit,
                        color="#FFE66D", linewidth=2.0, linestyle="--", zorder=4,
                        label=f"p = {lr.pvalue:.3f}")
                ax.legend(fontsize=9, framealpha=0.7, facecolor=_AXES_COLOR,
                          edgecolor=_SPINE_COLOR, labelcolor="white")
            ax.set_title("Cat 4+ Storm Fraction Trend", **_FONT_TITLE, pad=8)
            ax.tick_params(axis="x", labelrotation=45)

            # RI panel
            ax = axes[1]
            _style_ax(ax, xlabel="Decade", ylabel="RI Storm Fraction (%)")
            ax.bar(d_labels, decade_df["ri_fraction"],
                   color="#FF6060", alpha=0.85, edgecolor="white",
                   linewidth=0.4, zorder=3)
            if "ri_trend" in stat_results:
                lr_ri = stats.linregress(d_years, decade_df["ri_fraction"].values)
                x_pos = np.arange(len(d_labels))
                y_fit_ri = lr_ri.slope * d_years + lr_ri.intercept
                ax.plot(x_pos, y_fit_ri,
                        color="#FFE66D", linewidth=2.0, linestyle="--", zorder=4,
                        label=f"p = {lr_ri.pvalue:.3f}")
                ax.legend(fontsize=9, framealpha=0.7, facecolor=_AXES_COLOR,
                          edgecolor=_SPINE_COLOR, labelcolor="white")
            ax.set_title("Rapid Intensification Trend", **_FONT_TITLE, pad=8)
            ax.tick_params(axis="x", labelrotation=45)
        else:
            for ax in axes:
                ax.text(0.5, 0.5, "Insufficient data\n(< 3 decades)",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=14, color="#999999")

        fig.suptitle(f"{BASIN_NAMES.get(basin, basin)} — Cat 4+ and RI Trends",
                     fontsize=16, fontweight="bold", color="#E0E0E0", y=1.02)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(clim_dir / "cat4_and_ri_trends.png", dpi=200,
                    bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
    print(f"  Saved: cat4_and_ri_trends.png")


# ===================================================================
# Section 6: Per-Storm Summary Table
# ===================================================================

def run_summary_table(checkpoint_dir: Path, output_dir: Path,
                      index_df: pd.DataFrame, num_storms: int,
                      device: torch.device):
    """Compute per-storm error metrics for the selected test storms.

    Produces:
        - storm_summary.csv
        - storm_summary.png (formatted table image)
    """
    print("\n" + "=" * 70)
    print("  SECTION 6: Per-Storm Summary Table")
    print("=" * 70)

    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    # Select storms
    selected = _select_interesting_storms(index_df, num_storms=num_storms)
    if not selected:
        print("  No suitable test storms found.")
        return

    # Load models
    # Focus on the key models: Reg CNN, Reg LSTM, Reg Fusion, Cls CNN
    models_to_use = {
        "Reg CNN": MODEL_REGISTRY.get("Reg 4: CNN"),
        "Reg LSTM": MODEL_REGISTRY.get("Reg 2: LSTM"),
        "Reg Fusion": MODEL_REGISTRY.get("Reg 5: Fusion"),
        "Cls CNN": MODEL_REGISTRY.get("Cls 4: CNN"),
    }

    loaded = {}
    for short_name, info in models_to_use.items():
        if info is None:
            continue
        ckpt_path = checkpoint_dir / info["ckpt_pattern"]
        if not ckpt_path.exists():
            continue
        try:
            model = _load_checkpoint(info["model_cls"], ckpt_path, device)
            loaded[short_name] = {
                "model": model,
                "model_type": info["model_type"],
                "is_regression": info["is_regression"],
            }
            print(f"  Loaded {short_name}")
        except Exception as e:
            print(f"  Failed to load {short_name}: {e}")

    if not loaded:
        print("  No models loaded, skipping summary table.")
        return

    # Process each storm
    rows = []
    for year, storm_name in selected:
        storm_group = index_df[(index_df["year"] == year) &
                               (index_df["storm_name"] == storm_name)]
        if storm_group.empty:
            continue

        storm_data, env_data_list = _load_storm_data(storm_group, DATA_ROOT)
        if storm_data is None:
            continue

        actual_track = storm_data["positions"]
        wind_speeds = storm_data["wind_speeds"]
        T = len(actual_track)

        if T <= SEQ_LEN:
            continue

        # Prepare env data
        clean_env = [e if e is not None else _make_dummy_env_dict()
                     for e in env_data_list]

        peak_wind = float(wind_speeds.max())
        category = _saffir_category(peak_wind)
        duration_days = (T * 6) / 24

        row = {
            "storm_name": storm_name,
            "year": int(year),
            "duration_days": round(duration_days, 1),
            "peak_wind_ms": round(peak_wind, 1),
            "category": category,
            "n_timesteps": T,
        }

        # Compute mean track error for each model
        model_errors = {}
        for short_name, minfo in loaded.items():
            mtype = minfo["model_type"]
            is_reg = minfo["is_regression"]

            # Check data availability
            needs_env = mtype in ("env_single", "env_seq", "fusion")
            needs_3d = mtype in ("data3d", "fusion")
            env_available = sum(1 for e in env_data_list if e is not None)

            if needs_env and env_available < SEQ_LEN + 1:
                row[f"mae_km_{short_name}"] = None
                continue
            if needs_3d and not storm_data.get("data3d_tensors"):
                row[f"mae_km_{short_name}"] = None
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

                n_common = min(T, len(pred))
                if n_common > SEQ_LEN:
                    dists = [haversine_km(actual_track[i, 0], actual_track[i, 1],
                                          pred[i, 0], pred[i, 1])
                             for i in range(SEQ_LEN, n_common)]
                    mean_err = float(np.mean(dists))
                    row[f"mae_km_{short_name}"] = round(mean_err, 1)
                    model_errors[short_name] = mean_err
                else:
                    row[f"mae_km_{short_name}"] = None
            except Exception as e:
                row[f"mae_km_{short_name}"] = None
                print(f"    {storm_name}: {short_name} failed: {e}")

        # Determine best model
        if model_errors:
            best = min(model_errors, key=model_errors.get)
            row["best_model"] = best
        else:
            row["best_model"] = "N/A"

        rows.append(row)
        print(f"  {storm_name} ({year}): {model_errors}")

    if not rows:
        print("  No valid storms processed.")
        return

    # Save CSV
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(summary_dir / "storm_summary.csv", index=False)
    print(f"  Saved: storm_summary.csv ({len(rows)} storms)")

    # --- Formatted table image ---
    _render_summary_table(summary_df, summary_dir / "storm_summary.png", loaded)

    # Free GPU
    for minfo in loaded.values():
        del minfo["model"]
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


def _render_summary_table(df: pd.DataFrame, output_path: Path,
                          loaded_models: dict):
    """Render the summary DataFrame as a styled table image."""
    with plt.style.context("dark_background"):
        # Prepare display columns
        display_cols = ["storm_name", "year", "duration_days", "peak_wind_ms",
                        "category"]
        model_cols = [c for c in df.columns if c.startswith("mae_km_")]
        display_cols += model_cols + ["best_model"]

        # Clean column names for display
        col_labels = {
            "storm_name": "Storm",
            "year": "Year",
            "duration_days": "Days",
            "peak_wind_ms": "Peak (m/s)",
            "category": "Cat",
            "best_model": "Best",
        }
        for mc in model_cols:
            short = mc.replace("mae_km_", "")
            col_labels[mc] = f"{short}\n(km)"

        # Filter to available columns
        avail_cols = [c for c in display_cols if c in df.columns]
        tbl_data = df[avail_cols].copy()

        # Replace None/NaN with dash
        tbl_data = tbl_data.fillna("—")

        n_rows = len(tbl_data)
        n_cols = len(avail_cols)

        fig_width = max(14, n_cols * 1.8)
        fig_height = max(4, (n_rows + 1) * 0.45 + 1)

        fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor=_BG_COLOR)
        ax.axis("off")

        headers = [col_labels.get(c, c) for c in avail_cols]
        cell_text = tbl_data.values.tolist()

        table = ax.table(
            cellText=cell_text,
            colLabels=headers,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.4)

        # Style cells
        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor(_SPINE_COLOR)
            cell.set_linewidth(0.5)
            if r == 0:  # header
                cell.set_facecolor("#2A2D35")
                cell.set_text_props(color="white", fontweight="bold", fontsize=8)
            else:
                cell.set_facecolor(_AXES_COLOR)
                cell.set_text_props(color="#CCCCCC", fontsize=8)

                # Highlight best model column in green
                col_name = avail_cols[c] if c < len(avail_cols) else ""
                if col_name == "best_model":
                    cell.set_text_props(color="#4ECDC4", fontweight="bold",
                                        fontsize=8)

        ax.set_title("Per-Storm Track Error Summary (Test Set)",
                     fontsize=14, fontweight="bold", color="#E0E0E0", pad=20)

        plt.tight_layout()
        fig.savefig(output_path, dpi=200, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
    print(f"  Saved: storm_summary.png")


# ===================================================================
# Main entry point
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive post-training analysis for cyclone prediction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m src.scripts.run_analysis \\
        --results-dir results/20260325_233354 \\
        --output-dir results/analysis \\
        --num-storms 20

    python -m src.scripts.run_analysis \\
        --results-dir results/20260325_233354 \\
        --intensity-dir results/intensity_20260326_101551 \\
        --output-dir results/analysis
        """,
    )
    parser.add_argument(
        "--results-dir", type=str, required=True,
        help="Directory with track model checkpoints (from run_all.py)",
    )
    parser.add_argument(
        "--intensity-dir", type=str, default=None,
        help="Directory with intensity model checkpoints (from run_intensity.py) [optional]",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/analysis",
        help="Where to save all analysis outputs (default: results/analysis)",
    )
    parser.add_argument(
        "--num-storms", type=int, default=20,
        help="Number of test storms to visualize (default: 20)",
    )
    parser.add_argument(
        "--skip-tracks", action="store_true",
        help="Skip track prediction generation (Section 1)",
    )
    parser.add_argument(
        "--skip-residuals", action="store_true",
        help="Skip residual analysis (Sections 2-3)",
    )
    parser.add_argument(
        "--skip-cross-basin", action="store_true",
        help="Skip cross-basin intensity analysis (Section 4)",
    )
    parser.add_argument(
        "--skip-climate", action="store_true",
        help="Skip climate change analysis (Section 5)",
    )
    parser.add_argument(
        "--skip-summary", action="store_true",
        help="Skip per-storm summary table (Section 6)",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    checkpoint_dir = results_dir / "checkpoints"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        return
    if not checkpoint_dir.exists():
        print(f"ERROR: Checkpoints directory not found: {checkpoint_dir}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Results dir: {results_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Num storms: {args.num_storms}")

    # Load master index (used by multiple sections)
    if not MASTER_INDEX_PATH.exists():
        print(f"ERROR: Master index not found at {MASTER_INDEX_PATH}")
        print("  Run: python -m src.data.build_index --basin WP")
        return
    index_df = pd.read_csv(MASTER_INDEX_PATH)
    index_df = index_df[index_df["future_direction24"] >= 0]
    index_df = index_df[index_df["split"].isin(["train", "val", "test"])]
    print(f"Master index: {len(index_df)} samples")

    print("\n" + "#" * 70)
    print("#  COMPREHENSIVE POST-TRAINING ANALYSIS")
    print("#" * 70)

    # Section 1: Track predictions
    if not args.skip_tracks:
        try:
            run_track_predictions(results_dir, checkpoint_dir, output_dir,
                                  num_storms=args.num_storms)
        except Exception as e:
            print(f"  Section 1 FAILED: {e}")
            import traceback; traceback.print_exc()

    # Sections 2-3: Residuals and scatter (share model inference)
    residual_results = {}
    if not args.skip_residuals:
        try:
            residual_results = run_residual_analysis(
                checkpoint_dir, output_dir, index_df, device)
            if residual_results:
                run_scatter_analysis(residual_results, output_dir)
        except Exception as e:
            print(f"  Sections 2-3 FAILED: {e}")
            import traceback; traceback.print_exc()

    # Section 4: Cross-basin intensity
    if not args.skip_cross_basin:
        try:
            run_cross_basin_intensity(output_dir)
        except Exception as e:
            print(f"  Section 4 FAILED: {e}")
            import traceback; traceback.print_exc()

    # Section 5: Climate change
    if not args.skip_climate:
        try:
            run_climate_analysis(output_dir, basin="WP")
        except Exception as e:
            print(f"  Section 5 FAILED: {e}")
            import traceback; traceback.print_exc()

    # Section 6: Per-storm summary
    if not args.skip_summary:
        try:
            run_summary_table(checkpoint_dir, output_dir, index_df,
                              num_storms=args.num_storms, device=device)
        except Exception as e:
            print(f"  Section 6 FAILED: {e}")
            import traceback; traceback.print_exc()

    # Final report
    print("\n" + "#" * 70)
    print("#  ANALYSIS COMPLETE")
    print("#" * 70)
    print(f"\n  All outputs saved to: {output_dir}")
    print(f"  Subdirectories:")
    for subdir in sorted(output_dir.iterdir()):
        if subdir.is_dir():
            n_files = sum(1 for f in subdir.rglob("*") if f.is_file())
            print(f"    {subdir.name}/  ({n_files} files)")
    print()


if __name__ == "__main__":
    main()
