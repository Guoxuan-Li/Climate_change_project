"""
Intensity visualization: publication-quality plots and animated GIFs for
tropical cyclone intensity prediction analysis.

Provides:
- Time-series intensity evolution with Saffir-Simpson category bands
- 24-hour intensity change comparison (actual vs predicted)
- Decadal intensity statistics (multi-panel)
- Storm evolution animated GIFs with atmospheric fields
- Combined map + intensity inset animations

All geographic plots reuse the dark-background Cartopy style from
``trajectory_plots.py``.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from src.config import (
    DATA_ROOT, DATA1D_ROOT, DATA1D_COLS,
    MASTER_INDEX_PATH, NORM_LONG, NORM_LAT, NORM_WND,
    DATA3D_PRESSURE_LEVELS, PROJECT_ROOT,
)
from src.visualization.trajectory_plots import (
    _DarkMapStyle, norm_to_lon, norm_to_lat, norm_to_wind, _wind_colormap,
    _select_interesting_storms, HOURS_PER_STEP,
)

# ---------------------------------------------------------------------------
# Saffir-Simpson category definitions (m/s thresholds)
# ---------------------------------------------------------------------------

# Category name, lower threshold (m/s), colour
SAFFIR_SIMPSON = [
    ("TD",       0.0,  "#5EBAFF"),
    ("TS",      17.1,  "#00FAF4"),
    ("STS",     24.4,  "#FFFFCC"),
    ("TY",      32.6,  "#FFE775"),
    ("STY",     41.4,  "#FF8F20"),
    ("Super TY", 50.9, "#FF6060"),
]

# Threshold values for horizontal reference lines
_CAT_THRESHOLDS = {
    "TD":  17.1,
    "TS":  24.4,
    "STS": 32.6,
    "TY":  41.4,
    "STY": 50.9,
}

# Rapid intensification threshold: 30 kt / 24 h ~ 15.4 m/s per 24 h
RI_THRESHOLD_MS = 15.4

# Prediction line colours (colour-blind safe)
_PRED_LINE_COLORS = [
    "#FF6B6B",  # coral red
    "#4ECDC4",  # teal
    "#FFE66D",  # gold
    "#A8E6CF",  # mint
    "#FF8B94",  # salmon
    "#B8A9C9",  # lavender
]

# ---------------------------------------------------------------------------
# Shared typography and style helpers
# ---------------------------------------------------------------------------

_FONT_TITLE = {"fontsize": 14, "fontweight": "bold", "color": "#E0E0E0"}
_FONT_LABEL = {"fontsize": 11, "color": "#CCCCCC"}
_FONT_TICK  = {"labelsize": 9, "labelcolor": "#AAAAAA"}

_BG_COLOR   = "#0E1117"
_AXES_COLOR = "#1A1D23"
_GRID_COLOR = "#2A2D35"
_SPINE_COLOR = "#3A3D45"


def _style_ax(ax, xlabel="", ylabel="", title=""):
    """Apply publication-quality dark styling to a regular (non-Cartopy) axes."""
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


def _saffir_color(wind_ms: float) -> str:
    """Return the Saffir-Simpson category colour for a given wind speed."""
    for name, threshold, color in reversed(SAFFIR_SIMPSON):
        if wind_ms >= threshold:
            return color
    return SAFFIR_SIMPSON[0][2]


def _saffir_category(wind_ms: float) -> str:
    """Return the Saffir-Simpson category name for a given wind speed."""
    for name, threshold, _ in reversed(SAFFIR_SIMPSON):
        if wind_ms >= threshold:
            return name
    return "TD"


def _denorm_wind(wnd_norm):
    """Convert normalised wind to m/s (array-safe)."""
    return np.asarray(wnd_norm, dtype=float) * NORM_WND["scale"] + NORM_WND["offset"]


# ---------------------------------------------------------------------------
# 1. Intensity evolution time series
# ---------------------------------------------------------------------------

def plot_intensity_evolution(
    storm_name: str,
    year: int,
    actual_wind: np.ndarray,
    predicted_winds: dict[str, np.ndarray],
    output_path: str | Path,
):
    """Plot wind speed evolution over a storm's lifetime.

    Args:
        storm_name: e.g. "SURIGAE"
        year: e.g. 2021
        actual_wind: (T,) array of wind speed in m/s
        predicted_winds: dict mapping model_name -> (T,) array in m/s
        output_path: where to save the figure (PNG)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    T = len(actual_wind)
    hours = np.arange(T) * HOURS_PER_STEP

    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=_BG_COLOR)
        _style_ax(ax, xlabel="Hours from storm genesis",
                  ylabel="Wind speed (m/s)")

        # --- Category shading bands ---
        xlim = (hours[0] - 3, hours[-1] + 3)
        for i, (name, lo, color) in enumerate(SAFFIR_SIMPSON):
            hi = SAFFIR_SIMPSON[i + 1][1] if i + 1 < len(SAFFIR_SIMPSON) else 100
            ax.axhspan(lo, hi, color=color, alpha=0.07, zorder=0)

        # --- Category threshold dashed lines + right-side labels ---
        for cat_name, thresh in _CAT_THRESHOLDS.items():
            ax.axhline(thresh, color="#555555", linewidth=0.7, linestyle=":",
                       alpha=0.8, zorder=1)
            ax.annotate(
                cat_name, xy=(1.01, thresh), xycoords=("axes fraction", "data"),
                fontsize=7.5, color="#999999", va="center",
                annotation_clip=False,
            )

        # --- Actual wind: thick line coloured by category ---
        for i in range(T - 1):
            seg_color = _saffir_color(actual_wind[i])
            ax.plot(hours[i:i + 2], actual_wind[i:i + 2],
                    color=seg_color, linewidth=2.8, solid_capstyle="round",
                    zorder=5)
        # Scatter on top for dots
        colors = [_saffir_color(w) for w in actual_wind]
        ax.scatter(hours, actual_wind, c=colors, s=18, zorder=6,
                   edgecolors="white", linewidths=0.3, label="Actual")

        # --- Predicted winds ---
        for idx, (model_name, pred_w) in enumerate(predicted_winds.items()):
            c = _PRED_LINE_COLORS[idx % len(_PRED_LINE_COLORS)]
            ax.plot(np.arange(len(pred_w)) * HOURS_PER_STEP, pred_w,
                    color=c, linewidth=1.4, linestyle="--", alpha=0.85,
                    zorder=4, label=model_name)

        # --- Formatting ---
        ax.set_xlim(xlim)
        y_lo = max(0, actual_wind.min() - 8)
        y_hi = min(100, actual_wind.max() + 12)
        ax.set_ylim(y_lo, y_hi)

        # Title
        peak = actual_wind.max()
        cat = _saffir_category(peak)
        ax.set_title(
            f"{storm_name} ({year})  —  Peak: {peak:.1f} m/s ({cat})",
            **_FONT_TITLE, pad=12,
        )

        # Legend
        leg = ax.legend(loc="upper left", fontsize=8, framealpha=0.75,
                        facecolor=_AXES_COLOR, edgecolor=_SPINE_COLOR)
        for text in leg.get_texts():
            text.set_color("#DDDDDD")

        plt.tight_layout()
        fig.savefig(output_path, dpi=250, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
    print(f"  Saved intensity evolution: {output_path}")


# ---------------------------------------------------------------------------
# 2. Intensity change comparison (24 h delta)
# ---------------------------------------------------------------------------

def plot_intensity_change_comparison(
    storm_name: str,
    actual_delta: np.ndarray,
    predicted_deltas: dict[str, np.ndarray],
    output_path: str | Path,
):
    """Bar chart of 24-hour intensity change at each timestep.

    Args:
        storm_name: e.g. "SURIGAE"
        actual_delta: (N,) array of 24h wind change in m/s
        predicted_deltas: dict mapping model_name -> (N,) array in m/s
        output_path: where to save the figure
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    N = len(actual_delta)
    x = np.arange(N)

    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(10, 4.5), facecolor=_BG_COLOR)
        _style_ax(ax, xlabel="Timestep index (6 h intervals)",
                  ylabel=r"$\Delta$ Wind speed (m/s per 24 h)")

        # Colour bars by sign: red for intensifying, blue for weakening
        bar_colors = np.where(actual_delta >= 0, "#E74C3C", "#3498DB")
        bar_width = 0.6 / (1 + len(predicted_deltas))

        # Actual bars
        ax.bar(x - bar_width * len(predicted_deltas) / 2,
               actual_delta, width=bar_width, color=bar_colors,
               edgecolor="white", linewidth=0.3, alpha=0.85,
               label="Actual", zorder=3)

        # Predicted overlaid
        for idx, (model_name, pred_d) in enumerate(predicted_deltas.items()):
            c = _PRED_LINE_COLORS[idx % len(_PRED_LINE_COLORS)]
            offset = bar_width * (idx + 1) - bar_width * len(predicted_deltas) / 2
            n_plot = min(len(pred_d), N)
            ax.bar(x[:n_plot] + offset, pred_d[:n_plot],
                   width=bar_width, color=c, alpha=0.6,
                   edgecolor="white", linewidth=0.2,
                   label=model_name, zorder=3)

        # RI threshold band
        ax.axhline(RI_THRESHOLD_MS, color="#FF4444", linewidth=1.0,
                    linestyle="--", alpha=0.7, zorder=4)
        ax.axhline(-RI_THRESHOLD_MS, color="#4444FF", linewidth=1.0,
                    linestyle="--", alpha=0.7, zorder=4)
        ax.annotate("RI threshold", xy=(N - 0.5, RI_THRESHOLD_MS + 0.8),
                    fontsize=7.5, color="#FF6666", ha="right")

        # Shade RI events
        ri_mask = actual_delta >= RI_THRESHOLD_MS
        if ri_mask.any():
            for i in np.where(ri_mask)[0]:
                ax.axvspan(i - 0.45, i + 0.45, color="#FF4444", alpha=0.12,
                           zorder=0)

        ax.axhline(0, color="#666666", linewidth=0.6, zorder=1)

        ax.set_title(f"{storm_name} — 24 h Intensity Change", **_FONT_TITLE,
                     pad=10)

        leg = ax.legend(loc="upper left", fontsize=7.5, framealpha=0.75,
                        facecolor=_AXES_COLOR, edgecolor=_SPINE_COLOR)
        for t in leg.get_texts():
            t.set_color("#DDDDDD")

        plt.tight_layout()
        fig.savefig(output_path, dpi=250, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
    print(f"  Saved intensity change: {output_path}")


# ---------------------------------------------------------------------------
# 3. Decadal intensity statistics (4-panel)
# ---------------------------------------------------------------------------

def plot_decadal_intensity_stats(
    index_df: pd.DataFrame,
    output_path: str | Path,
):
    """Four-panel figure showing how cyclone intensity has changed over decades.

    Uses all Data1D files referenced in the master index (all splits).

    Args:
        index_df: master index DataFrame (must contain year, storm_name,
            data1d_file, basin columns)
        output_path: where to save the figure
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Collect per-storm statistics from Data1D ---
    storms = index_df.groupby(["year", "storm_name"]).first().reset_index()
    decade_stats = []

    for _, row in storms.iterrows():
        split_dir = row.get("split", "train")
        if split_dir == "unknown":
            split_dir = "train"
        d1d_path = DATA1D_ROOT / row["basin"] / split_dir / row["data1d_file"]
        if not d1d_path.exists():
            continue

        try:
            track_df = pd.read_csv(
                d1d_path, delimiter="\t", header=None, names=DATA1D_COLS,
                dtype={"timestamp": str},
            )
        except Exception:
            continue

        winds_ms = _denorm_wind(track_df["wnd_norm"].values)
        peak_wind = winds_ms.max()
        n_ts = len(winds_ms)

        # 24 h delta = wind[t+4] - wind[t]
        deltas_24h = []
        for t in range(n_ts - 4):
            deltas_24h.append(winds_ms[t + 4] - winds_ms[t])
        deltas_24h = np.array(deltas_24h) if deltas_24h else np.array([0.0])

        mean_intensification = deltas_24h.mean() if len(deltas_24h) > 0 else 0.0
        ri_count = int(np.sum(deltas_24h >= RI_THRESHOLD_MS))

        decade = (int(row["year"]) // 10) * 10
        decade_stats.append({
            "decade": decade,
            "year": int(row["year"]),
            "storm_name": row["storm_name"],
            "peak_wind": peak_wind,
            "mean_intensification": mean_intensification,
            "ri_count": ri_count,
            "is_cat4plus": peak_wind >= 41.4,
        })

    sdf = pd.DataFrame(decade_stats)
    if sdf.empty:
        print("  Warning: no Data1D files found for decadal stats.")
        return

    # Aggregate by decade
    decades = sorted(sdf["decade"].unique())
    # Filter out decades with very few storms
    decade_groups = sdf.groupby("decade")

    mean_peak = []
    frac_cat4 = []
    mean_intens = []
    ri_events = []
    decade_labels = []
    n_storms_per_decade = []

    for d in decades:
        g = decade_groups.get_group(d)
        n = len(g)
        if n < 5:
            continue
        decade_labels.append(f"{d}s")
        n_storms_per_decade.append(n)
        mean_peak.append(g["peak_wind"].mean())
        frac_cat4.append(g["is_cat4plus"].mean() * 100)
        mean_intens.append(g["mean_intensification"].mean())
        ri_events.append(g["ri_count"].sum())

    # --- Plot ---
    with plt.style.context("dark_background"):
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), facecolor=_BG_COLOR)

        bar_kw = dict(edgecolor="white", linewidth=0.4, zorder=3)

        # Panel 1: Mean peak wind speed
        ax = axes[0, 0]
        _style_ax(ax, ylabel="Wind speed (m/s)")
        colors1 = [_saffir_color(w) for w in mean_peak]
        ax.bar(decade_labels, mean_peak, color=colors1, alpha=0.85, **bar_kw)
        ax.set_title("Mean Peak Wind Speed", **_FONT_TITLE, pad=8)
        # Add value labels
        for i, v in enumerate(mean_peak):
            ax.text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=7.5,
                    color="#CCCCCC")

        # Panel 2: Fraction Cat4+
        ax = axes[0, 1]
        _style_ax(ax, ylabel="Fraction (%)")
        ax.bar(decade_labels, frac_cat4, color="#FF8F20", alpha=0.85, **bar_kw)
        ax.set_title("Storms Reaching STY+ (Cat 4+)", **_FONT_TITLE, pad=8)
        for i, v in enumerate(frac_cat4):
            ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=7.5,
                    color="#CCCCCC")

        # Panel 3: Mean 24h intensification rate
        ax = axes[1, 0]
        _style_ax(ax, ylabel="m/s per 24 h")
        bar_colors3 = ["#E74C3C" if v >= 0 else "#3498DB" for v in mean_intens]
        ax.bar(decade_labels, mean_intens, color=bar_colors3, alpha=0.85,
               **bar_kw)
        ax.axhline(0, color="#666666", linewidth=0.5)
        ax.set_title("Mean 24 h Intensification Rate", **_FONT_TITLE, pad=8)
        for i, v in enumerate(mean_intens):
            offset = 0.15 if v >= 0 else -0.4
            ax.text(i, v + offset, f"{v:+.2f}", ha="center", fontsize=7.5,
                    color="#CCCCCC")

        # Panel 4: Total RI events
        ax = axes[1, 1]
        _style_ax(ax, ylabel="Number of RI events")
        ax.bar(decade_labels, ri_events, color="#FF6060", alpha=0.85, **bar_kw)
        ax.set_title("Rapid Intensification Events", **_FONT_TITLE, pad=8)
        for i, v in enumerate(ri_events):
            ax.text(i, v + max(ri_events) * 0.01, str(v), ha="center",
                    fontsize=7.5, color="#CCCCCC")
        # Normalised RI per storm as secondary annotation
        ax2 = ax.twinx()
        ri_per_storm = [r / max(n, 1) for r, n in zip(ri_events, n_storms_per_decade)]
        ax2.plot(decade_labels, ri_per_storm, color="#FFE66D", linewidth=1.5,
                 marker="o", markersize=4, zorder=5, alpha=0.8)
        ax2.set_ylabel("RI per storm", fontsize=9, color="#FFE66D")
        ax2.tick_params(labelsize=8, labelcolor="#FFE66D", colors="#FFE66D")
        for spine in ax2.spines.values():
            spine.set_visible(False)

        fig.suptitle(
            "Western Pacific Tropical Cyclone Intensity — Decadal Trends (1950–2023)",
            fontsize=15, fontweight="bold", color="#E0E0E0", y=0.98,
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(output_path, dpi=250, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
    print(f"  Saved decadal intensity stats: {output_path}")


# ---------------------------------------------------------------------------
# 4. Storm animation (GIF)
# ---------------------------------------------------------------------------

def _load_wind_field_850(data_root, year, storm_name, timestamp):
    """Load 850 hPa wind speed from Data3D as (81, 81) array.

    850 hPa is pressure_level index 2 (levels: 200, 500, 850, 925).
    Wind speed = sqrt(u^2 + v^2).

    Returns None if file is missing or unreadable.
    """
    nc_path = (Path(data_root) / "Data3D" / "WP" / str(year) / storm_name /
               f"TCND_{storm_name}_{timestamp}_sst_z_u_v.nc")
    if not nc_path.exists():
        return None
    try:
        import xarray as xr
        ds = xr.open_dataset(str(nc_path))
        u = ds["u"].values[0, 2]   # (81, 81) at 850 hPa
        v = ds["v"].values[0, 2]
        ds.close()
        wspd = np.sqrt(u ** 2 + v ** 2)
        return wspd
    except Exception:
        return None


def create_storm_animation(
    storm_name: str,
    year: int,
    data_root: str | Path,
    index_df: pd.DataFrame,
    output_path: str | Path,
    fps: int = 4,
):
    """Create an animated GIF of a cyclone's evolution over time.

    Each frame shows:
    - Geographic map (Cartopy, dark background)
    - Storm position as a large coloured dot (colour = wind speed)
    - Track history up to that point (fading trail)
    - 850 hPa wind speed field as filled contour
    - Timestamp and wind speed / category text overlay

    Args:
        storm_name: e.g. "SURIGAE"
        year: e.g. 2021
        data_root: path to Data/ directory
        index_df: master index DataFrame
        output_path: where to save the GIF
        fps: frames per second
    """
    from matplotlib.animation import FuncAnimation, PillowWriter

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_root = Path(data_root)

    # --- Load storm track from Data1D ---
    storm_rows = index_df[(index_df["storm_name"] == storm_name) &
                          (index_df["year"] == year)]
    if storm_rows.empty:
        print(f"  Warning: {storm_name} ({year}) not found in index.")
        return

    first = storm_rows.iloc[0]
    split_dir = first.get("split", "test")
    if split_dir == "unknown":
        split_dir = "test"
    d1d_path = DATA1D_ROOT / first["basin"] / split_dir / first["data1d_file"]
    if not d1d_path.exists():
        print(f"  Warning: Data1D file not found: {d1d_path}")
        return

    track_df = pd.read_csv(
        d1d_path, delimiter="\t", header=None, names=DATA1D_COLS,
        dtype={"timestamp": str},
    )
    lons = np.array([norm_to_lon(v) for v in track_df["long_norm"]])
    lats = np.array([norm_to_lat(v) for v in track_df["lat_norm"]])
    winds = np.array([norm_to_wind(v) for v in track_df["wnd_norm"]])
    timestamps = track_df["timestamp"].values
    T = len(lons)

    # Limit frames for reasonable GIF size
    max_frames = 80
    stride = max(1, T // max_frames)
    frame_indices = list(range(0, T, stride))
    if frame_indices[-1] != T - 1:
        frame_indices.append(T - 1)

    # --- Map extent ---
    pad = 4.0
    extent = [
        lons.min() - pad, lons.max() + pad,
        lats.min() - pad, lats.max() + pad,
    ]

    # Wind speed colourmap
    cmap, norm = _wind_colormap()

    # Pre-load wind fields for all frames
    wind_fields = {}
    for fi in frame_indices:
        wf = _load_wind_field_850(data_root, year, storm_name, timestamps[fi])
        if wf is not None:
            wind_fields[fi] = wf

    # --- Build animation ---
    with plt.style.context("dark_background"):
        proj = ccrs.PlateCarree()
        fig, ax = plt.subplots(
            figsize=(7, 5.5), subplot_kw={"projection": proj},
            facecolor=_BG_COLOR,
        )
        fig.set_dpi(100)

        def _draw_frame(frame_idx_pos):
            """Draw a single animation frame."""
            ax.clear()
            fi = frame_indices[frame_idx_pos]

            _DarkMapStyle.dress_ax(ax)
            ax.set_extent(extent, crs=proj)

            # 850 hPa wind field contour
            if fi in wind_fields:
                wf = wind_fields[fi]
                # Grid: 81x81, 20 deg extent, centred on storm
                lon_c, lat_c = lons[fi], lats[fi]
                wf_lons = np.linspace(lon_c - 10, lon_c + 10, 81)
                wf_lats = np.linspace(lat_c - 10, lat_c + 10, 81)
                WF_LON, WF_LAT = np.meshgrid(wf_lons, wf_lats)
                levels = np.linspace(0, 50, 15)
                cf = ax.contourf(
                    WF_LON, WF_LAT, wf, levels=levels,
                    cmap="hot_r", alpha=0.35, transform=proj,
                    extend="max", zorder=2,
                )

            # Fading trail up to current frame
            trail_end = fi + 1
            if trail_end > 1:
                n_trail = trail_end
                alphas = np.linspace(0.15, 0.9, n_trail)
                for i in range(n_trail - 1):
                    seg_alpha = float(alphas[i])
                    seg_color = cmap(norm(winds[i]))
                    ax.plot(
                        lons[i:i + 2], lats[i:i + 2],
                        color=seg_color, linewidth=1.5, alpha=seg_alpha,
                        solid_capstyle="round", transform=proj, zorder=3,
                    )

            # Current position: large dot
            ax.scatter(
                [lons[fi]], [lats[fi]],
                c=[winds[fi]], cmap=cmap, norm=norm,
                s=160, edgecolors="white", linewidths=1.2,
                transform=proj, zorder=7,
            )

            # Text overlay
            ts = timestamps[fi]
            date_str = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]} {ts[8:10]}Z"
            cat = _saffir_category(winds[fi])
            info_text = (f"{storm_name} ({year})\n"
                         f"{date_str}\n"
                         f"{winds[fi]:.1f} m/s  ({cat})")
            ax.text(
                0.02, 0.98, info_text, transform=ax.transAxes,
                fontsize=9, fontweight="bold", color="white",
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#000000",
                          alpha=0.7, edgecolor="#555555"),
                zorder=10,
            )

            # Progress bar at bottom (using axes fraction coordinates)
            progress = (frame_idx_pos + 1) / len(frame_indices)
            ax.plot(
                [extent[0], extent[0] + (extent[1] - extent[0]) * progress],
                [extent[2], extent[2]],
                color="#FFE66D", linewidth=3, transform=proj,
                zorder=10, solid_capstyle="butt",
            )

        anim = FuncAnimation(
            fig, _draw_frame, frames=len(frame_indices),
            interval=1000 // fps, repeat=False,
        )
        writer = PillowWriter(fps=fps)
        anim.save(str(output_path), writer=writer,
                  savefig_kwargs={"facecolor": fig.get_facecolor()})
        plt.close(fig)
    print(f"  Saved storm animation: {output_path}")


# ---------------------------------------------------------------------------
# 5. Combined animation: map + intensity inset
# ---------------------------------------------------------------------------

def create_intensity_comparison_animation(
    storm_name: str,
    year: int,
    data_root: str | Path,
    index_df: pd.DataFrame,
    actual_wind: np.ndarray,
    predicted_winds: dict[str, np.ndarray],
    output_path: str | Path,
    fps: int = 4,
):
    """Animated GIF with storm map and real-time intensity inset.

    Main panel: geographic map with storm position and 850 hPa wind field.
    Inset panel: intensity time series drawn progressively (actual vs predicted).

    Args:
        storm_name: e.g. "SURIGAE"
        year: e.g. 2021
        data_root: path to Data/ directory
        index_df: master index DataFrame
        actual_wind: (T,) array of wind speed in m/s
        predicted_winds: dict mapping model_name -> (T,) array in m/s
        output_path: where to save the GIF
        fps: frames per second
    """
    from matplotlib.animation import FuncAnimation, PillowWriter

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_root = Path(data_root)

    # --- Load storm track ---
    storm_rows = index_df[(index_df["storm_name"] == storm_name) &
                          (index_df["year"] == year)]
    if storm_rows.empty:
        print(f"  Warning: {storm_name} ({year}) not found in index.")
        return

    first = storm_rows.iloc[0]
    split_dir = first.get("split", "test")
    if split_dir == "unknown":
        split_dir = "test"
    d1d_path = DATA1D_ROOT / first["basin"] / split_dir / first["data1d_file"]
    if not d1d_path.exists():
        print(f"  Warning: Data1D file not found: {d1d_path}")
        return

    track_df = pd.read_csv(
        d1d_path, delimiter="\t", header=None, names=DATA1D_COLS,
        dtype={"timestamp": str},
    )
    lons = np.array([norm_to_lon(v) for v in track_df["long_norm"]])
    lats = np.array([norm_to_lat(v) for v in track_df["lat_norm"]])
    winds_track = np.array([norm_to_wind(v) for v in track_df["wnd_norm"]])
    timestamps = track_df["timestamp"].values
    T = len(lons)

    # Align actual_wind length with track
    T_wind = min(T, len(actual_wind))

    # Frame indices
    max_frames = 80
    stride = max(1, T_wind // max_frames)
    frame_indices = list(range(0, T_wind, stride))
    if frame_indices[-1] != T_wind - 1:
        frame_indices.append(T_wind - 1)

    # Map extent
    pad = 4.0
    extent = [lons.min() - pad, lons.max() + pad,
              lats.min() - pad, lats.max() + pad]

    cmap, wnorm = _wind_colormap()

    # Pre-load wind fields
    wind_fields = {}
    for fi in frame_indices:
        wf = _load_wind_field_850(data_root, year, storm_name, timestamps[fi])
        if wf is not None:
            wind_fields[fi] = wf

    # Intensity plot bounds
    all_winds = [actual_wind[:T_wind]]
    for pw in predicted_winds.values():
        all_winds.append(pw[:T_wind])
    all_concat = np.concatenate(all_winds)
    y_lo = max(0, np.nanmin(all_concat) - 8)
    y_hi = min(100, np.nanmax(all_concat) + 10)
    hours = np.arange(T_wind) * HOURS_PER_STEP

    # --- Build animation ---
    with plt.style.context("dark_background"):
        fig = plt.figure(figsize=(10, 6), facecolor=_BG_COLOR, dpi=100)
        gs = gridspec.GridSpec(1, 2, width_ratios=[1.4, 1], wspace=0.02,
                               figure=fig)

        # Map axes (left)
        proj = ccrs.PlateCarree()
        ax_map = fig.add_subplot(gs[0], projection=proj)

        # Inset axes (right)
        ax_inset = fig.add_subplot(gs[1])

        def _draw_frame(frame_idx_pos):
            fi = frame_indices[frame_idx_pos]

            # --- Map panel ---
            ax_map.clear()
            _DarkMapStyle.dress_ax(ax_map)
            ax_map.set_extent(extent, crs=proj)

            # Wind field contour
            if fi in wind_fields:
                wf = wind_fields[fi]
                lon_c, lat_c = lons[fi], lats[fi]
                wf_lons = np.linspace(lon_c - 10, lon_c + 10, 81)
                wf_lats = np.linspace(lat_c - 10, lat_c + 10, 81)
                WF_LON, WF_LAT = np.meshgrid(wf_lons, wf_lats)
                levels = np.linspace(0, 50, 15)
                ax_map.contourf(
                    WF_LON, WF_LAT, wf, levels=levels,
                    cmap="hot_r", alpha=0.35, transform=proj,
                    extend="max", zorder=2,
                )

            # Fading trail
            trail_end = fi + 1
            if trail_end > 1:
                alphas = np.linspace(0.15, 0.9, trail_end)
                for i in range(trail_end - 1):
                    seg_color = cmap(wnorm(winds_track[i]))
                    ax_map.plot(
                        lons[i:i + 2], lats[i:i + 2],
                        color=seg_color, linewidth=1.5,
                        alpha=float(alphas[i]),
                        solid_capstyle="round", transform=proj, zorder=3,
                    )

            # Current position
            ax_map.scatter(
                [lons[fi]], [lats[fi]],
                c=[winds_track[fi]], cmap=cmap, norm=wnorm,
                s=140, edgecolors="white", linewidths=1.0,
                transform=proj, zorder=7,
            )

            # Info text
            ts = timestamps[fi]
            date_str = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]} {ts[8:10]}Z"
            cat = _saffir_category(winds_track[fi])
            ax_map.text(
                0.02, 0.98,
                f"{storm_name} ({year})\n{date_str}\n"
                f"{winds_track[fi]:.1f} m/s ({cat})",
                transform=ax_map.transAxes, fontsize=8, fontweight="bold",
                color="white", verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#000000",
                          alpha=0.7, edgecolor="#555555"),
                zorder=10,
            )

            # --- Inset: intensity time series ---
            ax_inset.clear()
            ax_inset.set_facecolor(_AXES_COLOR)
            ax_inset.tick_params(**_FONT_TICK, direction="in", length=3,
                                width=0.5, colors="#666666")
            for spine in ax_inset.spines.values():
                spine.set_color(_SPINE_COLOR)
                spine.set_linewidth(0.5)
            ax_inset.grid(True, color=_GRID_COLOR, linewidth=0.3, alpha=0.5)

            # Category thresholds
            for cat_name, thresh in _CAT_THRESHOLDS.items():
                ax_inset.axhline(thresh, color="#444444", linewidth=0.5,
                                 linestyle=":", alpha=0.6)

            # Draw actual wind up to current frame
            end = fi + 1
            if end > 1:
                for i in range(end - 1):
                    seg_c = _saffir_color(actual_wind[i])
                    ax_inset.plot(
                        hours[i:i + 2], actual_wind[i:i + 2],
                        color=seg_c, linewidth=2.0, solid_capstyle="round",
                    )
                ax_inset.scatter(
                    [hours[fi]], [actual_wind[fi]],
                    c=[_saffir_color(actual_wind[fi])],
                    s=40, edgecolors="white", linewidths=0.5, zorder=6,
                )

            # Draw predicted winds up to current frame
            for pidx, (mname, pw) in enumerate(predicted_winds.items()):
                pc = _PRED_LINE_COLORS[pidx % len(_PRED_LINE_COLORS)]
                pw_end = min(end, len(pw))
                if pw_end > 1:
                    ax_inset.plot(
                        hours[:pw_end], pw[:pw_end],
                        color=pc, linewidth=1.2, linestyle="--",
                        alpha=0.8, label=mname if frame_idx_pos == 0 else "",
                    )

            ax_inset.set_xlim(hours[0] - 3, hours[T_wind - 1] + 3)
            ax_inset.set_ylim(y_lo, y_hi)
            ax_inset.set_xlabel("Hours", fontsize=8, color="#AAAAAA")
            ax_inset.set_ylabel("Wind (m/s)", fontsize=8, color="#AAAAAA")
            ax_inset.set_title("Intensity", fontsize=10, fontweight="bold",
                               color="#E0E0E0", pad=5)

            # Vertical "now" line
            ax_inset.axvline(hours[fi], color="#FFE66D", linewidth=0.8,
                             alpha=0.6, linestyle="-")

        anim = FuncAnimation(
            fig, _draw_frame, frames=len(frame_indices),
            interval=1000 // fps, repeat=False,
        )
        writer = PillowWriter(fps=fps)
        anim.save(str(output_path), writer=writer,
                  savefig_kwargs={"facecolor": fig.get_facecolor()})
        plt.close(fig)
    print(f"  Saved intensity comparison animation: {output_path}")


# ---------------------------------------------------------------------------
# 6. Main entry point
# ---------------------------------------------------------------------------

def generate_intensity_visualizations(
    results_dir: str | Path,
    index_path: str | Path,
    data_root: str | Path,
    num_storms: int = 6,
):
    """Generate all intensity visualizations.

    1. Loads the master index.
    2. Selects the same 6 test storms used for trajectory visualization.
    3. For each storm: generates intensity evolution plot, storm animation,
       and comparison animation.
    4. Generates the decadal intensity statistics plot (all WP storms).
    5. Saves everything to results_dir/intensity_viz/.

    Args:
        results_dir: root results directory (e.g. results/20260325_233354)
        index_path: path to master_index_WP.csv
        data_root: path to Data/ directory
        num_storms: number of test storms to visualize
    """
    results_dir = Path(results_dir)
    index_path = Path(index_path)
    data_root = Path(data_root)

    out_dir = results_dir / "intensity_viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Generating intensity visualizations")
    print("=" * 60)

    # Load index
    df = pd.read_csv(index_path, dtype={"timestamp": str})
    print(f"  Loaded index: {len(df)} rows, "
          f"{df.groupby(['year', 'storm_name']).ngroups} storms")

    # --- Decadal statistics (all WP storms) ---
    print("\n--- Decadal intensity statistics ---")
    plot_decadal_intensity_stats(df, out_dir / "decadal_intensity_stats.png")

    # --- Select test storms ---
    selected = _select_interesting_storms(df, num_storms=num_storms)
    if not selected:
        print("  No suitable test storms found.")
        return
    print(f"\n--- Selected {len(selected)} test storms ---")
    for yr, sname in selected:
        print(f"  - {sname} ({yr})")

    # --- Per-storm visualizations ---
    for yr, sname in selected:
        print(f"\n--- {sname} ({yr}) ---")

        # Load Data1D
        storm_rows = df[(df["storm_name"] == sname) & (df["year"] == yr)]
        if storm_rows.empty:
            continue

        first = storm_rows.iloc[0]
        split_dir = first.get("split", "test")
        if split_dir == "unknown":
            split_dir = "test"
        d1d_path = DATA1D_ROOT / first["basin"] / split_dir / first["data1d_file"]
        if not d1d_path.exists():
            print(f"  Data1D not found: {d1d_path}")
            continue

        track_df = pd.read_csv(
            d1d_path, delimiter="\t", header=None, names=DATA1D_COLS,
            dtype={"timestamp": str},
        )
        winds_ms = _denorm_wind(track_df["wnd_norm"].values)
        T = len(winds_ms)

        # For demonstration: use empty predicted_winds dict
        # (In production, load model predictions here)
        predicted_winds = {}

        # 1. Intensity evolution plot
        safe_name = sname.replace(" ", "_").replace("-", "_")
        plot_intensity_evolution(
            sname, yr, winds_ms, predicted_winds,
            out_dir / f"intensity_{safe_name}_{yr}.png",
        )

        # 2. Intensity change comparison
        if T > 4:
            delta_24h = np.array([winds_ms[t + 4] - winds_ms[t]
                                  for t in range(T - 4)])
            plot_intensity_change_comparison(
                sname, delta_24h, {},
                out_dir / f"delta_{safe_name}_{yr}.png",
            )

        # 3. Storm animation (GIF)
        create_storm_animation(
            sname, yr, data_root, df,
            out_dir / f"anim_{safe_name}_{yr}.gif",
            fps=4,
        )

        # 4. Combined animation with intensity inset
        create_intensity_comparison_animation(
            sname, yr, data_root, df,
            winds_ms, predicted_winds,
            out_dir / f"anim_intensity_{safe_name}_{yr}.gif",
            fps=4,
        )

    print("\n" + "=" * 60)
    print(f"All intensity visualizations saved to: {out_dir}")
    print("=" * 60)
