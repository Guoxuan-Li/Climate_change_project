"""
Generalization experiments:
  1. Temporal shift: train on different historical periods, test on 2017-2021
  2. Cross-basin: train on WP, evaluate zero-shot on other basins

Uses Reg 2 (LSTM on Data1D) and Reg 4 (CNN on Data3D) — best lightweight
and best overall models.

Usage:
    python -m src.scripts.run_generalization --epochs 100
    python -m src.scripts.run_generalization --temporal-only --epochs 50
    python -m src.scripts.run_generalization --basin-only --epochs 50
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import (
    MASTER_INDEX_PATH, DATA_ROOT, DATA1D_ROOT, ENV_DATA_ROOT,
    BATCH_SIZE, EPOCHS, LR, PATIENCE, SEED, NUM_DIRECTION_CLASSES,
    PROJECT_ROOT, INDEX_DIR, DIRECTION_LABELS,
)
from src.training.trainer import RegressionTrainer, set_seed, get_device
from src.training.evaluate import (
    compute_regression_metrics, print_regression_metrics,
    evaluate_regression_model,
)


# ── Helpers ────────────────────────────────────────────────────────────────

def save_metrics(metrics, path):
    serializable = {}
    for k, v in metrics.items():
        if isinstance(v, np.floating):
            serializable[k] = float(v)
        elif isinstance(v, np.ndarray):
            serializable[k] = v.tolist()
        else:
            serializable[k] = v
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)


def train_reg_lstm(train_df, val_df, test_df, args, checkpoint_dir, name):
    """Train a RegLSTM and evaluate on test set."""
    from src.data.regression_dataset import RegData1DSequenceDataset
    from src.models.regression_models import RegLSTM

    train_ds = RegData1DSequenceDataset(train_df, DATA_ROOT)
    val_ds = RegData1DSequenceDataset(val_df, DATA_ROOT)
    test_ds = RegData1DSequenceDataset(test_df, DATA_ROOT)

    print(f"    Samples: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
        print(f"    Skipping {name}: insufficient data")
        return None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False,
                             num_workers=4, pin_memory=True)

    model = RegLSTM()
    criterion = torch.nn.MSELoss()
    trainer = RegressionTrainer(
        model=model, criterion=criterion,
        checkpoint_dir=checkpoint_dir,
        experiment_name=name,
    )
    trainer.train(train_loader, val_loader, epochs=args.epochs, patience=args.patience)
    return trainer.evaluate_on_test(test_loader)


def train_reg_cnn(train_df, val_df, test_df, args, checkpoint_dir, name):
    """Train a RegCNN3D and evaluate on test set."""
    from src.data.regression_dataset import RegData3DDataset
    from src.models.regression_models import RegCNN3D

    # Filter to samples with Data3D
    train_df = train_df[train_df["data3d_exists"] == True]
    val_df = val_df[val_df["data3d_exists"] == True]
    test_df = test_df[test_df["data3d_exists"] == True]

    train_ds = RegData3DDataset(train_df, DATA_ROOT)
    val_ds = RegData3DDataset(val_df, DATA_ROOT)
    test_ds = RegData3DDataset(test_df, DATA_ROOT)

    print(f"    Samples: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
        print(f"    Skipping {name}: insufficient data")
        return None

    bs = max(args.batch_size // 2, 16)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False,
                             num_workers=4, pin_memory=True)

    model = RegCNN3D()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr * 0.1, weight_decay=1e-4)
    trainer = RegressionTrainer(
        model=model, criterion=criterion, optimizer=optimizer,
        checkpoint_dir=checkpoint_dir,
        experiment_name=name,
    )
    trainer.train(train_loader, val_loader, epochs=args.epochs, patience=args.patience)
    return trainer.evaluate_on_test(test_loader)


def eval_pretrained_model(model_cls_path, checkpoint_path, test_df, args, model_needs_3d=False):
    """Load a pretrained model and evaluate on a test dataframe."""
    import importlib

    device = get_device()

    # Instantiate model
    module_path, cls_name = model_cls_path.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    model = getattr(mod, cls_name)()

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # Build dataset
    if model_needs_3d:
        from src.data.regression_dataset import RegData3DDataset
        test_df = test_df[test_df["data3d_exists"] == True]
        test_ds = RegData3DDataset(test_df, DATA_ROOT)
    else:
        from src.data.regression_dataset import RegData1DSequenceDataset
        test_ds = RegData1DSequenceDataset(test_df, DATA_ROOT)

    if len(test_ds) == 0:
        return None

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)
    return evaluate_regression_model(model, test_loader, str(device))


# ── Experiment 1: Temporal shift ───────────────────────────────────────────

def run_temporal_experiment(args, results_dir):
    """Train on different historical cutoffs, test on 2017-2021.

    Cutoffs: 1950-1990, 1950-2000, 1950-2010, 1950-2016 (original)
    Validation: last 20% of the training period in each case.
    Test: always 2017-2021 (the original test set).
    """
    print(f"\n{'#'*60}")
    print("  EXPERIMENT 1: TEMPORAL GENERALIZATION")
    print(f"{'#'*60}")

    df = pd.read_csv(MASTER_INDEX_PATH)
    df = df[df["delta_lon_norm"].notna() & df["delta_lat_norm"].notna()]
    df = df[df["split"].isin(["train", "val", "test"])]

    # The test set is always 2017-2021
    test_df = df[df["year"] >= 2017]
    print(f"\n  Test set (2017-2021): {len(test_df):,} samples")

    cutoffs = [1990, 2000, 2010, 2016]
    results = {}

    for cutoff in cutoffs:
        print(f"\n{'='*60}")
        print(f"  Training period: 1950-{cutoff}")
        print(f"{'='*60}")

        # All data from 1950 to cutoff
        period_df = df[(df["year"] <= cutoff) & (df["split"].isin(["train", "val"]))]

        if len(period_df) < 100:
            print(f"  Too few samples ({len(period_df)}), skipping")
            continue

        # Split: 80% train, 20% val (by year within the period)
        years = sorted(period_df["year"].unique())
        val_cutoff = years[int(len(years) * 0.8)]
        train_df = period_df[period_df["year"] <= val_cutoff]
        val_df = period_df[period_df["year"] > val_cutoff]

        print(f"  Train: {len(train_df):,} samples (1950-{val_cutoff})")
        print(f"  Val:   {len(val_df):,} samples ({val_cutoff+1}-{cutoff})")
        print(f"  Test:  {len(test_df):,} samples (2017-2021)")

        period_results = {}

        # Reg LSTM
        print(f"\n  --- Reg LSTM (1950-{cutoff}) ---")
        lstm_metrics = train_reg_lstm(
            train_df, val_df, test_df, args,
            checkpoint_dir=results_dir / "checkpoints",
            name=f"temporal_lstm_{cutoff}",
        )
        if lstm_metrics:
            period_results["Reg LSTM"] = lstm_metrics
            save_metrics(lstm_metrics, results_dir / f"temporal_lstm_{cutoff}.json")

        # Reg CNN (only if enough Data3D samples)
        print(f"\n  --- Reg CNN (1950-{cutoff}) ---")
        cnn_metrics = train_reg_cnn(
            train_df, val_df, test_df, args,
            checkpoint_dir=results_dir / "checkpoints",
            name=f"temporal_cnn_{cutoff}",
        )
        if cnn_metrics:
            period_results["Reg CNN"] = cnn_metrics
            save_metrics(cnn_metrics, results_dir / f"temporal_cnn_{cutoff}.json")

        results[cutoff] = period_results

    # Build comparison table
    print(f"\n{'='*60}")
    print("  TEMPORAL GENERALIZATION RESULTS")
    print(f"{'='*60}")

    rows = []
    for cutoff in cutoffs:
        if cutoff not in results:
            continue
        for model_name, m in results[cutoff].items():
            rows.append({
                "Training Period": f"1950-{cutoff}",
                "Model": model_name,
                "MAE (km)": f"{m['mae_km']:.1f}",
                "Median (km)": f"{m['median_track_error_km']:.1f}",
                "Dir Acc": f"{m['derived_accuracy']:.4f}",
                "Dir F1": f"{m['derived_macro_f1']:.4f}",
                "R² lon": f"{m['r2_dlon']:.4f}",
                "R² lat": f"{m['r2_dlat']:.4f}",
            })

    table_df = pd.DataFrame(rows)
    print(table_df.to_string(index=False))
    table_df.to_csv(results_dir / "temporal_comparison.csv", index=False)

    # Plot: MAE vs training cutoff
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for model_name, color, marker in [("Reg LSTM", "#FF6B6B", "o"), ("Reg CNN", "#4ECDC4", "s")]:
        cutoff_vals = []
        mae_vals = []
        acc_vals = []
        for cutoff in cutoffs:
            if cutoff in results and model_name in results[cutoff]:
                m = results[cutoff][model_name]
                cutoff_vals.append(cutoff)
                mae_vals.append(m["mae_km"])
                acc_vals.append(m["derived_accuracy"])

        if cutoff_vals:
            axes[0].plot(cutoff_vals, mae_vals, f"-{marker}", color=color,
                         label=model_name, linewidth=2, markersize=8)
            axes[1].plot(cutoff_vals, acc_vals, f"-{marker}", color=color,
                         label=model_name, linewidth=2, markersize=8)

    axes[0].set_xlabel("Training data cutoff year")
    axes[0].set_ylabel("Mean Track Error (km)")
    axes[0].set_title("24h Track Error vs Training Period")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].invert_yaxis()  # lower is better

    axes[1].set_xlabel("Training data cutoff year")
    axes[1].set_ylabel("Derived Direction Accuracy")
    axes[1].set_title("Direction Accuracy vs Training Period")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle("Temporal Generalization: Does older training data degrade predictions?",
                 fontweight="bold")
    plt.tight_layout()
    fig.savefig(results_dir / "temporal_generalization.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved temporal_generalization.png")

    return results


# ── Experiment 2: Cross-basin generalization ───────────────────────────────

def run_basin_experiment(args, results_dir):
    """Evaluate WP-trained models on all 6 basins (zero-shot transfer).

    Uses the best checkpoints from the main run to avoid retraining.
    Also builds indexes for non-WP basins on the fly.
    """
    print(f"\n{'#'*60}")
    print("  EXPERIMENT 2: CROSS-BASIN GENERALIZATION")
    print(f"{'#'*60}")

    # First, check for existing WP-trained checkpoints
    # Look in any results directory for the best models
    all_results_dirs = sorted(PROJECT_ROOT.glob("results/*/checkpoints"))
    lstm_ckpt = None
    cnn_ckpt = None
    for rd in reversed(all_results_dirs):  # most recent first
        if lstm_ckpt is None and (rd / "reg_stage2_lstm_1d_best.pt").exists():
            lstm_ckpt = rd / "reg_stage2_lstm_1d_best.pt"
        if cnn_ckpt is None and (rd / "reg_stage4_cnn_3d_best.pt").exists():
            cnn_ckpt = rd / "reg_stage4_cnn_3d_best.pt"

    if lstm_ckpt is None and cnn_ckpt is None:
        print("  No pretrained WP models found. Run the main pipeline first.")
        return {}

    print(f"  Using LSTM checkpoint: {lstm_ckpt}")
    print(f"  Using CNN checkpoint:  {cnn_ckpt}")

    basins = ["WP", "EP", "NA", "NI", "SI", "SP"]
    results = {}

    for basin in basins:
        print(f"\n{'='*60}")
        print(f"  Basin: {basin}")
        print(f"{'='*60}")

        # Build or load index for this basin
        index_path = INDEX_DIR / f"master_index_{basin}.csv"
        if not index_path.exists():
            print(f"  Building index for {basin}...")
            from src.data.build_index import build_index_for_basin
            basin_df = build_index_for_basin(basin)
            if basin_df.empty:
                print(f"  No data for {basin}, skipping.")
                continue
            basin_df.to_csv(index_path, index=False)
        else:
            basin_df = pd.read_csv(index_path)

        # Filter for regression-valid test samples
        # For non-WP basins, use the "test" split if available, otherwise use all data
        basin_df = basin_df[basin_df["delta_lon_norm"].notna() & basin_df["delta_lat_norm"].notna()]

        if "test" in basin_df["split"].values:
            test_basin_df = basin_df[basin_df["split"] == "test"]
        else:
            # No pre-defined test split — use storms from 2017+ as test
            test_basin_df = basin_df[basin_df["year"] >= 2017]

        if len(test_basin_df) == 0:
            print(f"  No test samples for {basin}, skipping.")
            continue

        print(f"  Test samples: {len(test_basin_df):,}")

        basin_results = {}

        # Evaluate LSTM
        if lstm_ckpt:
            print(f"  Evaluating Reg LSTM on {basin}...")
            lstm_m = eval_pretrained_model(
                "src.models.regression_models.RegLSTM",
                lstm_ckpt, test_basin_df, args, model_needs_3d=False,
            )
            if lstm_m:
                basin_results["Reg LSTM"] = lstm_m
                print_regression_metrics(lstm_m, f"Reg LSTM on {basin}")

        # Evaluate CNN
        if cnn_ckpt:
            print(f"  Evaluating Reg CNN on {basin}...")
            cnn_m = eval_pretrained_model(
                "src.models.regression_models.RegCNN3D",
                cnn_ckpt, test_basin_df, args, model_needs_3d=True,
            )
            if cnn_m:
                basin_results["Reg CNN"] = cnn_m
                print_regression_metrics(cnn_m, f"Reg CNN on {basin}")

        results[basin] = basin_results

    # Build comparison table
    print(f"\n{'='*60}")
    print("  CROSS-BASIN GENERALIZATION RESULTS")
    print(f"{'='*60}")

    rows = []
    for basin in basins:
        if basin not in results:
            continue
        for model_name, m in results[basin].items():
            rows.append({
                "Basin": basin,
                "Model": model_name,
                "MAE (km)": f"{m['mae_km']:.1f}",
                "Median (km)": f"{m['median_track_error_km']:.1f}",
                "Dir Acc": f"{m['derived_accuracy']:.4f}",
                "Dir F1": f"{m['derived_macro_f1']:.4f}",
                "R² lon": f"{m['r2_dlon']:.4f}",
                "R² lat": f"{m['r2_dlat']:.4f}",
                "Training Basin": "WP" if basin != "WP" else "WP (in-domain)",
            })

    table_df = pd.DataFrame(rows)
    print(table_df.to_string(index=False))
    table_df.to_csv(results_dir / "basin_comparison.csv", index=False)

    # Plot: grouped bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    basin_order = [b for b in basins if b in results]
    x = np.arange(len(basin_order))
    width = 0.35

    for ax_idx, (metric, ylabel, title, invert) in enumerate([
        ("mae_km", "Mean Track Error (km)", "24h Track Error by Basin", True),
        ("derived_accuracy", "Derived Direction Accuracy", "Direction Accuracy by Basin", False),
    ]):
        lstm_vals = []
        cnn_vals = []
        for basin in basin_order:
            lstm_vals.append(results[basin].get("Reg LSTM", {}).get(metric, 0))
            cnn_vals.append(results[basin].get("Reg CNN", {}).get(metric, 0))

        ax = axes[ax_idx]
        bars1 = ax.bar(x - width/2, lstm_vals, width, label="Reg LSTM", color="#FF6B6B", alpha=0.85)
        bars2 = ax.bar(x + width/2, cnn_vals, width, label="Reg CNN", color="#4ECDC4", alpha=0.85)

        ax.set_xlabel("Basin")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(basin_order)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        # Annotate bars
        for bars in [bars1, bars2]:
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax.annotate(f"{h:.0f}" if metric == "mae_km" else f"{h:.2f}",
                                xy=(bar.get_x() + bar.get_width()/2, h),
                                xytext=(0, 3), textcoords="offset points",
                                ha="center", va="bottom", fontsize=8)

        # Highlight WP as the in-domain basin
        wp_idx = basin_order.index("WP") if "WP" in basin_order else -1
        if wp_idx >= 0:
            ax.axvspan(wp_idx - 0.5, wp_idx + 0.5, alpha=0.1, color="gold")
            ax.text(wp_idx, ax.get_ylim()[1] * 0.95, "in-domain",
                    ha="center", fontsize=8, style="italic", color="gold")

    plt.suptitle("Cross-Basin Generalization: WP-trained models on all basins",
                 fontweight="bold")
    plt.tight_layout()
    fig.savefig(results_dir / "basin_generalization.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved basin_generalization.png")

    return results


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generalization experiments")
    parser.add_argument("--temporal-only", action="store_true")
    parser.add_argument("--basin-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    set_seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = PROJECT_ROOT / "results" / f"generalization_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "checkpoints").mkdir(exist_ok=True)

    config = vars(args).copy()
    config["timestamp"] = timestamp
    config["device"] = str(get_device())
    with open(results_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n  Results directory: {results_dir}")

    run_temporal = not args.basin_only
    run_basin = not args.temporal_only

    t_start = time.time()

    if run_temporal:
        run_temporal_experiment(args, results_dir)

    if run_basin:
        run_basin_experiment(args, results_dir)

    total_time = time.time() - t_start
    print(f"\n{'#'*60}")
    print(f"  GENERALIZATION EXPERIMENTS COMPLETE")
    print(f"  Total time: {total_time/60:.1f} min")
    print(f"  Results: {results_dir}")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()
