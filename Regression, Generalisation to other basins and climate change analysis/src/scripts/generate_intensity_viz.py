"""
Generate complete intensity visualizations including model predictions.

Produces:
  - Confusion matrices for classification models
  - Comparison bar charts (cls + reg)
  - Per-storm intensity evolution with model predictions overlaid
  - Per-storm 24h intensity change with model predictions
  - Animated GIFs with model predictions
  - Decadal statistics

Usage:
    python -m src.scripts.generate_intensity_viz --intensity-dir results/intensity_20260326_101551
"""
import argparse
import json
import importlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import torch
import seaborn as sns

from src.config import (
    MASTER_INDEX_PATH, DATA_ROOT, DATA1D_ROOT, DATA1D_COLS,
    DATA1D_FEATURE_COLS, SEQ_LEN, INTENSITY_LABELS, NUM_INTENSITY_CLASSES,
    NORM_WND, PROJECT_ROOT,
)
from src.data.utils import env_dict_to_vector
from src.training.evaluate import compute_intensity_cls_metrics

# ── Saffir-Simpson palette ─────────────────────────────────────────────────
_SS_THRESHOLDS = [0, 17.1, 24.4, 32.6, 41.4, 50.9, 100]
_SS_LABELS = ["TD", "TS", "STS", "TY", "STY", "Super TY"]
_SS_COLORS = ["#5EBAFF", "#00FAF4", "#FFFFCC", "#FFE775", "#FF8F20", "#FF4040"]

def _ss_category(wind_ms):
    for i, thresh in enumerate(_SS_THRESHOLDS[1:]):
        if wind_ms < thresh:
            return i, _SS_LABELS[i], _SS_COLORS[i]
    return 5, _SS_LABELS[5], _SS_COLORS[5]

def _wind_to_color(wind_ms):
    _, _, c = _ss_category(wind_ms)
    return c

# ── Model loading ──────────────────────────────────────────────────────────

INTENSITY_MODEL_REGISTRY = {
    "Cls MLP": {
        "ckpt": "int_cls_mlp_best.pt",
        "cls": "src.models.intensity_models.IntensityClsMLP",
        "type": "env_single", "is_reg": False,
    },
    "Cls LSTM": {
        "ckpt": "int_cls_lstm_best.pt",
        "cls": "src.models.intensity_models.IntensityClsLSTM",
        "type": "data1d_seq", "is_reg": False,
    },
    "Cls Env Temp": {
        "ckpt": "int_cls_env_temporal_best.pt",
        "cls": "src.models.intensity_models.IntensityClsEnvTemporal",
        "type": "env_seq", "is_reg": False,
    },
    "Cls CNN": {
        "ckpt": "int_cls_cnn_best.pt",
        "cls": "src.models.intensity_models.IntensityClsCNN",
        "type": "data3d", "is_reg": False,
    },
    "Reg MLP": {
        "ckpt": "int_reg_mlp_best.pt",
        "cls": "src.models.intensity_models.IntensityRegMLP",
        "type": "env_single", "is_reg": True,
    },
    "Reg LSTM": {
        "ckpt": "int_reg_lstm_best.pt",
        "cls": "src.models.intensity_models.IntensityRegLSTM",
        "type": "data1d_seq", "is_reg": True,
    },
    "Reg Env Temp": {
        "ckpt": "int_reg_env_temporal_best.pt",
        "cls": "src.models.intensity_models.IntensityRegEnvTemporal",
        "type": "env_seq", "is_reg": True,
    },
    "Reg CNN": {
        "ckpt": "int_reg_cnn_best.pt",
        "cls": "src.models.intensity_models.IntensityRegCNN",
        "type": "data3d", "is_reg": True,
    },
}

def _load_model(cls_path, ckpt_path, device):
    mod_path, cls_name = cls_path.rsplit(".", 1)
    mod = importlib.import_module(mod_path)
    model = getattr(mod, cls_name)()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


# ── Per-storm prediction ───────────────────────────────────────────────────

def _load_storm(storm_group, data_root):
    """Load Data1D features, wind speeds, and Env-Data for a storm."""
    storm_group = storm_group.sort_values("timestamp").reset_index(drop=True)
    first = storm_group.iloc[0]

    d1d_path = DATA1D_ROOT / first["basin"] / first["split"] / first["data1d_file"]
    if not d1d_path.exists():
        return None

    track_df = pd.read_csv(d1d_path, delimiter="\t", header=None,
                           names=DATA1D_COLS, dtype={"timestamp": str})
    track_df["timestamp"] = track_df["timestamp"].astype(str).str.strip()

    features = track_df[DATA1D_FEATURE_COLS].values.astype(np.float32)
    wind_norm = track_df["wnd_norm"].values.astype(np.float32)
    wind_ms = wind_norm * 25.0 + 40.0
    timestamps = track_df["timestamp"].values

    # Env-Data
    ts_to_env = {}
    for _, row in storm_group.iterrows():
        ts_to_env[str(row["timestamp"]).strip()] = Path(data_root) / row["env_path"]

    env_list = []
    for ts in timestamps:
        ts_str = str(ts).strip()
        if ts_str in ts_to_env and ts_to_env[ts_str].exists():
            env_list.append(np.load(str(ts_to_env[ts_str]), allow_pickle=True).item())
        else:
            env_list.append(None)

    # Data3D paths
    ts_to_d3d = {}
    for _, row in storm_group.iterrows():
        if row.get("data3d_exists", False) and row.get("data3d_path", ""):
            ts_to_d3d[str(row["timestamp"]).strip()] = row["data3d_path"]

    return {
        "features": features,
        "wind_norm": wind_norm,
        "wind_ms": wind_ms,
        "timestamps": timestamps,
        "env_list": env_list,
        "ts_to_d3d": ts_to_d3d,
        "basin": first["basin"],
        "year": first["year"],
        "storm_name": first["storm_name"],
    }


def _predict_intensity_for_storm(model, storm, device, model_type, is_reg):
    """Predict intensity at each timestep for one storm.

    For regression: returns predicted delta_wnd_norm per step.
    For classification: returns predicted class per step.

    Then converts to predicted wind speed time series.
    """
    features = storm["features"]
    wind_norm = storm["wind_norm"]
    wind_ms = storm["wind_ms"]
    env_list = storm["env_list"]
    ts_to_d3d = storm["ts_to_d3d"]
    timestamps = storm["timestamps"]
    T = len(features)

    predicted_wind_ms = np.full(T, np.nan)
    predicted_wind_ms[:SEQ_LEN] = wind_ms[:SEQ_LEN]  # first SEQ_LEN are given

    # Dummy env dict for None entries
    dummy_env = {
        "area": np.zeros(6, dtype=np.float32), "wind": 0.0,
        "intensity_class": np.zeros(6, dtype=np.float32), "move_velocity": 0.0,
        "month": np.zeros(12, dtype=np.float32),
        "location_long": np.zeros(36, dtype=np.float32),
        "location_lat": np.zeros(12, dtype=np.float32),
        "history_direction12": -1, "history_direction24": -1,
    }

    with torch.no_grad():
        for t in range(SEQ_LEN, T):
            try:
                if model_type == "data1d_seq":
                    seq = features[t - SEQ_LEN:t]
                    x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
                    out = model(x)

                elif model_type == "env_single":
                    env = env_list[t] if env_list[t] is not None else dummy_env
                    vec = env_dict_to_vector(env)
                    x = torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(device)
                    out = model(x)

                elif model_type == "env_seq":
                    vecs = []
                    for j in range(t - SEQ_LEN, t):
                        env = env_list[j] if env_list[j] is not None else dummy_env
                        vecs.append(env_dict_to_vector(env))
                    x = torch.tensor(np.stack(vecs), dtype=torch.float32).unsqueeze(0).to(device)
                    out = model(x)

                elif model_type == "data3d":
                    ts_str = str(timestamps[t]).strip()
                    if ts_str not in ts_to_d3d:
                        predicted_wind_ms[t] = wind_ms[t]
                        continue
                    from src.visualization.trajectory_plots import _load_data3d_tensor
                    d3d = _load_data3d_tensor(
                        DATA_ROOT, storm["basin"], storm["year"],
                        storm["storm_name"], ts_str)
                    if d3d is None:
                        predicted_wind_ms[t] = wind_ms[t]
                        continue
                    x = d3d.unsqueeze(0).to(device)
                    out = model(x)
                else:
                    continue

                if is_reg:
                    delta_norm = out.cpu().item()
                    # Predicted wind at t = actual wind at t-4 + predicted delta
                    if t >= 4:
                        predicted_wind_ms[t] = wind_ms[t - 4] + delta_norm * 25.0
                    else:
                        predicted_wind_ms[t] = wind_ms[t]
                else:
                    pred_class = out.argmax(dim=1).item()
                    # Map class to approximate wind change
                    # 0=strengthening(+5), 1=str-then-weak(0), 2=weakening(-5), 3=maintaining(0)
                    delta_map = {0: 5.0, 1: 0.0, 2: -5.0, 3: 0.0}
                    delta_ms = delta_map[pred_class]
                    predicted_wind_ms[t] = wind_ms[t - 1] + delta_ms

            except Exception:
                predicted_wind_ms[t] = wind_ms[t]

    # Clip to physical range
    predicted_wind_ms = np.clip(predicted_wind_ms, 0, 120)
    return predicted_wind_ms


# ── Visualization functions ────────────────────────────────────────────────

def plot_confusion_matrices(results_dir, viz_dir):
    """Plot confusion matrices for all classification models."""
    cls_files = sorted(results_dir.glob("int_cls_*_test.json"))
    if not cls_files:
        return

    fig, axes = plt.subplots(1, len(cls_files), figsize=(5 * len(cls_files), 4.5))
    if len(cls_files) == 1:
        axes = [axes]

    for ax, fpath in zip(axes, cls_files):
        with open(fpath) as f:
            m = json.load(f)
        cm = np.array(m["confusion_matrix"])
        name = fpath.stem.replace("int_cls_", "").replace("_test", "").upper()

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=INTENSITY_LABELS, yticklabels=INTENSITY_LABELS,
                    ax=ax)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("True", fontsize=9)
        ax.set_title(f"Cls {name}\nAcc={m['accuracy']:.3f}  F1={m['macro_f1']:.3f}",
                     fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=7)

    plt.suptitle("Intensity Classification — Confusion Matrices", fontweight="bold")
    plt.tight_layout()
    fig.savefig(viz_dir / "confusion_matrices.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved confusion_matrices.png")


def plot_comparison_charts(results_dir, viz_dir):
    """Bar charts comparing all intensity models."""
    # Classification
    cls_models = []
    for fpath in sorted(results_dir.glob("int_cls_*_test.json")):
        with open(fpath) as f:
            m = json.load(f)
        name = fpath.stem.replace("int_cls_", "").replace("_test", "")
        cls_models.append({"name": f"Cls {name}", **m})

    # Add persistence
    pers_path = results_dir / "baseline_persistence.json"
    if pers_path.exists():
        with open(pers_path) as f:
            pm = json.load(f)
        cls_models.insert(0, {"name": "Persistence", **pm})

    # Regression
    reg_models = []
    for fpath in sorted(results_dir.glob("int_reg_*_test.json")):
        with open(fpath) as f:
            m = json.load(f)
        name = fpath.stem.replace("int_reg_", "").replace("_test", "")
        reg_models.append({"name": f"Reg {name}", **m})

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Classification accuracy
    if cls_models:
        names = [m["name"] for m in cls_models]
        accs = [m["accuracy"] for m in cls_models]
        colors = ["#4ECDC4" if "Persistence" in n else "#FF6B6B" for n in names]
        bars = axes[0].barh(names, accs, color=colors, alpha=0.85)
        for bar, v in zip(bars, accs):
            axes[0].text(v + 0.01, bar.get_y() + bar.get_height()/2,
                        f"{v:.3f}", va="center", fontsize=9)
        axes[0].set_xlabel("Accuracy")
        axes[0].set_title("Classification Accuracy", fontweight="bold")
        axes[0].set_xlim(0, 0.8)
        axes[0].grid(axis="x", alpha=0.3)

    # Panel 2: Classification macro F1
    if cls_models:
        f1s = [m["macro_f1"] for m in cls_models]
        bars = axes[1].barh(names, f1s, color=colors, alpha=0.85)
        for bar, v in zip(bars, f1s):
            axes[1].text(v + 0.01, bar.get_y() + bar.get_height()/2,
                        f"{v:.3f}", va="center", fontsize=9)
        axes[1].set_xlabel("Macro F1")
        axes[1].set_title("Classification Macro F1", fontweight="bold")
        axes[1].set_xlim(0, 0.8)
        axes[1].grid(axis="x", alpha=0.3)

    # Panel 3: Regression MAE (m/s)
    if reg_models:
        rnames = [m["name"] for m in reg_models]
        maes = [m["mae_ms"] for m in reg_models]
        bars = axes[2].barh(rnames, maes, color="#FFE66D", alpha=0.85)
        for bar, v in zip(bars, maes):
            axes[2].text(v + 0.05, bar.get_y() + bar.get_height()/2,
                        f"{v:.2f}", va="center", fontsize=9)
        axes[2].set_xlabel("MAE (m/s)")
        axes[2].set_title("Regression MAE", fontweight="bold")
        axes[2].grid(axis="x", alpha=0.3)

    plt.suptitle("Intensity Prediction — Model Comparison (WP Test Set 2017-2021)",
                 fontweight="bold", fontsize=13)
    plt.tight_layout()
    fig.savefig(viz_dir / "intensity_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved intensity_comparison.png")


def plot_storm_intensity_with_predictions(storm, predictions, viz_dir):
    """Plot intensity evolution with model predictions overlaid."""
    name = storm["storm_name"]
    year = storm["year"]
    wind_ms = storm["wind_ms"]
    T = len(wind_ms)
    hours = np.arange(T) * 6

    peak_wind = wind_ms.max()
    _, peak_cat, _ = _ss_category(peak_wind)

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    # Category shading
    for i in range(len(_SS_THRESHOLDS) - 1):
        ax.axhspan(_SS_THRESHOLDS[i], _SS_THRESHOLDS[i+1],
                   alpha=0.15, color=_SS_COLORS[i])
    # Category labels on right
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim() if ax.get_ylim()[1] > 0 else (0, 80))
    ax2.set_yticks([(a+b)/2 for a, b in zip(_SS_THRESHOLDS[:-1], _SS_THRESHOLDS[1:])])
    ax2.set_yticklabels(_SS_LABELS, fontsize=8, color="#aaaaaa")
    ax2.tick_params(right=False)

    # Actual wind — colored by category
    for i in range(T - 1):
        ax.plot(hours[i:i+2], wind_ms[i:i+2],
                color=_wind_to_color(wind_ms[i]), linewidth=2.5, solid_capstyle="round")
    ax.scatter(hours, wind_ms, c=[_wind_to_color(w) for w in wind_ms],
              s=25, zorder=5, edgecolors="white", linewidths=0.3, label="Actual")

    # Model predictions
    pred_colors = ["#FF6B6B", "#4ECDC4", "#FFE66D", "#A8E6CF", "#FF8B94", "#B8A9C9"]
    for idx, (mname, pred_wind) in enumerate(predictions.items()):
        color = pred_colors[idx % len(pred_colors)]
        valid = ~np.isnan(pred_wind)
        ax.plot(hours[valid], pred_wind[valid], "--", color=color,
                linewidth=1.2, alpha=0.9, label=mname)

    ax.set_xlabel("Hours from storm genesis", color="white", fontsize=10)
    ax.set_ylabel("Wind speed (m/s)", color="white", fontsize=10)
    ax.set_title(f"{name} ({year})  —  Peak: {peak_wind:.0f} m/s ({peak_cat})",
                fontsize=13, fontweight="bold", color="white")
    ax.set_ylim(0, max(80, peak_wind + 10))
    ax2.set_ylim(ax.get_ylim())
    ax.tick_params(colors="white")
    ax.grid(alpha=0.15, color="white")

    leg = ax.legend(loc="upper right", fontsize=8, framealpha=0.7,
                   facecolor="#2a2a2a", edgecolor="#555555")
    for text in leg.get_texts():
        text.set_color("white")

    safe = f"{name}_{year}".replace(" ", "_").replace("-", "_")
    out = viz_dir / f"intensity_pred_{safe}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved {out.name}")


def create_intensity_gif(storm, predictions, viz_dir, data_root, fps=4):
    """Animated GIF: map + intensity time series with predictions."""
    import matplotlib.animation as animation
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import xarray as xr

    name = storm["storm_name"]
    year = storm["year"]
    wind_ms = storm["wind_ms"]
    features = storm["features"]
    timestamps = storm["timestamps"]
    ts_to_d3d = storm["ts_to_d3d"]
    T = len(wind_ms)
    hours = np.arange(T) * 6

    # Positions
    lon = (features[:, 0] * 50 + 1800) / 10.0
    lat = features[:, 1] * 50 / 10.0

    # Stride for reasonable GIF size (~2-4MB per GIF)
    max_frames = 40
    stride = max(1, T // max_frames)
    frame_indices = list(range(0, T, stride))

    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(14, 6), dpi=150)
    fig.patch.set_facecolor("#1a1a2e")

    # Left: map — extent from actual track only
    ax_map = fig.add_axes([0.02, 0.05, 0.55, 0.85], projection=proj)
    ax_map.set_facecolor("#1a1a2e")
    land = cfeature.NaturalEarthFeature("physical", "land", "50m", facecolor="#16213e")
    ax_map.add_feature(land, zorder=1)
    ax_map.coastlines(linewidth=0.5, color="#666666", zorder=2)
    pad = 5
    extent = [float(lon.min()-pad), float(lon.max()+pad),
              float(lat.min()-pad), float(lat.max()+pad)]
    ax_map.set_extent(extent, crs=proj)
    gl = ax_map.gridlines(draw_labels=True, linewidth=0.3, alpha=0.3, color="#555555")
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"color": "#aaaaaa", "size": 8}

    # Right: intensity chart
    ax_int = fig.add_axes([0.62, 0.12, 0.35, 0.75])
    ax_int.set_facecolor("#1a1a2e")
    ax_int.set_xlim(0, hours[-1])
    ax_int.set_ylim(0, max(80, wind_ms.max() + 10))
    ax_int.set_xlabel("Hours", color="white", fontsize=9)
    ax_int.set_ylabel("Wind (m/s)", color="white", fontsize=9)
    ax_int.set_title("Intensity", color="white", fontsize=11, fontweight="bold")
    ax_int.tick_params(colors="white")
    ax_int.grid(alpha=0.15, color="white")
    # Category bands
    for i in range(len(_SS_THRESHOLDS) - 1):
        ax_int.axhspan(_SS_THRESHOLDS[i], _SS_THRESHOLDS[i+1],
                      alpha=0.1, color=_SS_COLORS[i])

    # Text overlay
    txt = ax_map.text(0.02, 0.95, "", transform=ax_map.transAxes, color="white",
                     fontsize=11, fontweight="bold", va="top",
                     bbox=dict(facecolor="black", alpha=0.7, edgecolor="none"))

    # Dynamic artists
    trail_line, = ax_map.plot([], [], "-", color="white", linewidth=1, alpha=0.5,
                              transform=proj, zorder=3)
    storm_dot, = ax_map.plot([], [], "o", markersize=12, markeredgecolor="white",
                             markeredgewidth=1.5, transform=proj, zorder=7)
    actual_line, = ax_int.plot([], [], "-", color="white", linewidth=2, label="Actual")

    pred_colors = ["#FF6B6B", "#4ECDC4", "#FFE66D"]
    pred_lines = {}
    pred_names = list(predictions.keys())[:3]  # max 3 for readability
    for idx, mname in enumerate(pred_names):
        ln, = ax_int.plot([], [], "--", color=pred_colors[idx], linewidth=1.2,
                         alpha=0.9, label=mname)
        pred_lines[mname] = ln

    ax_int.legend(loc="upper left", fontsize=7, framealpha=0.5,
                 facecolor="#2a2a2e", edgecolor="none", labelcolor="white")

    # Wind field mesh
    mesh_holder = [None]

    def animate(frame_idx):
        t = frame_indices[frame_idx]

        # Update trail
        trail_line.set_data(lon[:t+1], lat[:t+1])

        # Storm dot
        color = _wind_to_color(wind_ms[t])
        storm_dot.set_data([lon[t]], [lat[t]])
        storm_dot.set_color(color)

        # Text
        _, cat, _ = _ss_category(wind_ms[t])
        ts = str(timestamps[t])
        date_str = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]} {ts[8:]}Z"
        txt.set_text(f"{name} ({year})\n{date_str}\n{wind_ms[t]:.0f} m/s  ({cat})")

        # Wind field
        if mesh_holder[0] is not None:
            mesh_holder[0].remove()
            mesh_holder[0] = None

        ts_str = str(timestamps[t]).strip()
        if ts_str in ts_to_d3d:
            nc_path = Path(data_root) / ts_to_d3d[ts_str]
            if nc_path.exists():
                try:
                    ds = xr.open_dataset(str(nc_path))
                    u850 = ds["u"].values[0, 2]  # pressure_level=2 = 850hPa
                    v850 = ds["v"].values[0, 2]
                    wspd = np.sqrt(u850**2 + v850**2)
                    lo = ds["longitude"].values
                    la = ds["latitude"].values
                    lo2d, la2d = np.meshgrid(lo, la)
                    ds.close()
                    mesh_holder[0] = ax_map.pcolormesh(
                        lo2d, la2d, wspd, cmap="YlOrRd", vmin=0, vmax=50,
                        alpha=0.6, shading="auto", transform=proj, zorder=2)
                except Exception:
                    pass

        # Intensity chart
        actual_line.set_data(hours[:t+1], wind_ms[:t+1])
        for mname in pred_names:
            pw = predictions[mname]
            valid_mask = ~np.isnan(pw[:t+1])
            pred_lines[mname].set_data(hours[:t+1][valid_mask], pw[:t+1][valid_mask])

        return [trail_line, storm_dot, txt, actual_line] + list(pred_lines.values())

    anim = animation.FuncAnimation(fig, animate, frames=len(frame_indices),
                                   interval=1000/fps, blit=False)
    safe = f"{name}_{year}".replace(" ", "_").replace("-", "_")
    out = viz_dir / f"anim_intensity_{safe}.gif"
    anim.save(str(out), writer="pillow", fps=fps, dpi=120)
    plt.close(fig)
    print(f"  Saved {out.name}")


# ── Storm selection ────────────────────────────────────────────────────────

def _select_storms(df, num=6):
    test_df = df[(df["split"] == "test") & (df["future_direction24"] >= 0)]
    storms = test_df.groupby(["year", "storm_name"]).size().reset_index(name="count")
    storms = storms[storms["count"] >= 20].sort_values("count", ascending=False)
    selected = []
    for _, row in storms.iterrows():
        selected.append((row["year"], row["storm_name"]))
        if len(selected) >= num:
            break
    return selected


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--intensity-dir", type=str, required=True,
                        help="Path to intensity results directory")
    parser.add_argument("--num-storms", type=int, default=6)
    parser.add_argument("--best-n", type=int, default=3,
                        help="Number of best models to show in per-storm plots")
    args = parser.parse_args()

    results_dir = Path(args.intensity_dir)
    ckpt_dir = results_dir / "checkpoints"
    viz_dir = results_dir / "intensity_viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(MASTER_INDEX_PATH)
    print(f"\n{'='*60}")
    print("  INTENSITY VISUALIZATION")
    print(f"{'='*60}")

    # 1. Confusion matrices
    print("\n--- Confusion matrices ---")
    plot_confusion_matrices(results_dir, viz_dir)

    # 2. Comparison charts
    print("\n--- Comparison charts ---")
    plot_comparison_charts(results_dir, viz_dir)

    # 3. Decadal stats
    print("\n--- Decadal stats ---")
    from src.visualization.intensity_plots import plot_decadal_intensity_stats
    plot_decadal_intensity_stats(df, viz_dir / "decadal_intensity_stats.png")

    # 4. Load models
    print("\n--- Loading models ---")
    loaded = {}
    for mname, info in INTENSITY_MODEL_REGISTRY.items():
        ckpt_path = ckpt_dir / info["ckpt"]
        if not ckpt_path.exists():
            continue
        try:
            model = _load_model(info["cls"], ckpt_path, device)
            loaded[mname] = {"model": model, "type": info["type"], "is_reg": info["is_reg"]}
            print(f"  Loaded {mname}")
        except Exception as e:
            print(f"  Failed {mname}: {e}")

    if not loaded:
        print("  No models loaded, skipping per-storm plots.")
        return

    # 5. Per-storm visualizations
    selected = _select_storms(df, args.num_storms)
    print(f"\n--- Selected {len(selected)} test storms ---")

    for year, storm_name in selected:
        print(f"\n  {storm_name} ({year})")
        group = df[(df["year"] == year) & (df["storm_name"] == storm_name)]
        storm = _load_storm(group, DATA_ROOT)
        if storm is None:
            print("    Could not load, skipping.")
            continue

        # Predict with each model
        predictions = {}
        for mname, minfo in loaded.items():
            pred = _predict_intensity_for_storm(
                minfo["model"], storm, device, minfo["type"], minfo["is_reg"])
            predictions[mname] = pred

        # Pick best N models by MAE against actual
        model_errors = {}
        for mname, pred in predictions.items():
            valid = ~np.isnan(pred) & ~np.isnan(storm["wind_ms"])
            if valid.sum() > SEQ_LEN:
                mae = np.mean(np.abs(pred[valid] - storm["wind_ms"][valid]))
                model_errors[mname] = mae
        best_names = sorted(model_errors, key=model_errors.get)[:args.best_n]
        best_preds = {k: predictions[k] for k in best_names}

        # Static intensity plot
        plot_storm_intensity_with_predictions(storm, best_preds, viz_dir)

        # Animated GIF
        create_intensity_gif(storm, best_preds, viz_dir, DATA_ROOT, fps=4)

    print(f"\n{'='*60}")
    print(f"  All saved to: {viz_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
