"""
Intensity temporal generalization experiment.

Trains the best intensity models (CNN cls + CNN reg) on 4 historical cutoffs
(1950-1990, 1950-2000, 1950-2010, 1950-2016), tests on 2017-2021.

Produces comparison table and plot (MAE vs cutoff year).

KEY HYPOTHESIS: if climate change is increasing rapid intensification,
models trained on older data should underpredict intensity changes in
recent storms.

Usage:
    python -m src.scripts.run_intensity_temporal --epochs 50
    python -m src.scripts.run_intensity_temporal --epochs 100 --models cls_cnn reg_cnn
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
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.config import (
    MASTER_INDEX_PATH, BATCH_SIZE, EPOCHS, LR, FOCAL_GAMMA, PATIENCE,
    NUM_INTENSITY_CLASSES, DATA_ROOT, INTENSITY_LABELS, SEED, PROJECT_ROOT,
)
from src.data.utils import compute_class_weights
from src.training.losses import FocalLoss
from src.training.trainer import set_seed, get_device
from src.training.evaluate import (
    compute_intensity_cls_metrics, print_intensity_cls_metrics,
    evaluate_intensity_cls_model,
    compute_intensity_reg_metrics, print_intensity_reg_metrics,
    evaluate_intensity_reg_model,
)


# ── Helpers ────────────────────────────────────────────────────────────────

def save_metrics(metrics, path):
    serializable = {}
    for k, v in metrics.items():
        if isinstance(v, np.floating):
            serializable[k] = float(v)
        elif isinstance(v, np.integer):
            serializable[k] = int(v)
        elif isinstance(v, np.ndarray):
            serializable[k] = v.tolist()
        else:
            serializable[k] = v
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)


def make_weighted_sampler(labels, num_classes):
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weight_per_class = 1.0 / counts
    sample_weights = weight_per_class[labels]
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(labels),
        replacement=True,
    )


# ── Training Functions ────────────────────────────────────────────────────

def train_intensity_cls_cnn(train_df, val_df, test_df, args, checkpoint_dir, name):
    """Train an IntensityClsCNN and evaluate on test set."""
    from src.data.intensity_dataset import IntensityClsData3DDataset
    from src.models.intensity_models import IntensityClsCNN

    train_df = train_df[train_df["data3d_exists"] == True]
    val_df = val_df[val_df["data3d_exists"] == True]
    test_df = test_df[test_df["data3d_exists"] == True]

    train_ds = IntensityClsData3DDataset(train_df, DATA_ROOT)
    val_ds = IntensityClsData3DDataset(val_df, DATA_ROOT)
    test_ds = IntensityClsData3DDataset(test_df, DATA_ROOT)

    print(f"    Samples: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
        print(f"    Skipping {name}: insufficient data")
        return None

    # Extract labels for weighting
    train_labels = train_ds.df["future_inte_change24"].values.astype(np.int64)
    class_weights = compute_class_weights(train_labels, NUM_INTENSITY_CLASSES)
    sampler = make_weighted_sampler(train_labels, NUM_INTENSITY_CLASSES)

    bs = max(args.batch_size // 2, 16)
    train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False,
                             num_workers=4, pin_memory=True)

    model = IntensityClsCNN()
    criterion = FocalLoss(gamma=FOCAL_GAMMA, weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr * 0.1, weight_decay=1e-4)

    device = get_device()
    model = model.to(device)
    criterion = criterion.to(device)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    # Training loop (simplified, inline)
    best_f1 = 0.0
    best_epoch = 0
    patience_counter = 0

    print(f"\n{'='*60}")
    print(f"  Training (Intensity Cls CNN): {name}")
    print(f"  Device: {device}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            features, targets = batch
            features = features.to(device)
            targets = targets.to(device)
            logits = model(features)
            loss = criterion(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        train_loss = total_loss / max(n_batches, 1)

        # Validate
        val_metrics = evaluate_intensity_cls_model(model, val_loader, str(device))
        val_f1 = val_metrics["macro_f1"]
        scheduler.step(val_f1)

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch:3d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} | "
              f"val_acc={val_metrics['accuracy']:.4f} | "
              f"val_f1={val_f1:.4f} | "
              f"lr={lr:.2e} | {elapsed:.1f}s")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            ckpt_path = checkpoint_dir / f"{name}_best.pt"
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                         "best_metric": best_f1}, ckpt_path)
            print(f"    -> New best! F1={best_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n  Early stopping (best epoch: {best_epoch})")
                break

    # Load best
    ckpt_path = checkpoint_dir / f"{name}_best.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    # Test
    metrics = evaluate_intensity_cls_model(model, test_loader, str(device))
    print_intensity_cls_metrics(metrics, f"Test: {name}")
    return metrics


def train_intensity_reg_cnn(train_df, val_df, test_df, args, checkpoint_dir, name):
    """Train an IntensityRegCNN and evaluate on test set."""
    from src.data.intensity_dataset import IntensityRegData3DDataset
    from src.models.intensity_models import IntensityRegCNN

    train_df = train_df[train_df["data3d_exists"] == True]
    val_df = val_df[val_df["data3d_exists"] == True]
    test_df = test_df[test_df["data3d_exists"] == True]

    train_ds = IntensityRegData3DDataset(train_df, DATA_ROOT)
    val_ds = IntensityRegData3DDataset(val_df, DATA_ROOT)
    test_ds = IntensityRegData3DDataset(test_df, DATA_ROOT)

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

    model = IntensityRegCNN()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr * 0.1, weight_decay=1e-4)

    device = get_device()
    model = model.to(device)
    criterion = criterion.to(device)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    print(f"\n{'='*60}")
    print(f"  Training (Intensity Reg CNN): {name}")
    print(f"  Device: {device}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            features, targets = batch
            features = features.to(device)
            targets = targets.to(device)
            preds = model(features)
            loss = criterion(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        train_loss = total_loss / max(n_batches, 1)

        # Validate
        model.eval()
        val_loss = 0.0
        vn = 0
        with torch.no_grad():
            for batch in val_loader:
                features, targets = batch
                features = features.to(device)
                targets = targets.to(device)
                preds = model(features)
                loss = criterion(preds, targets)
                val_loss += loss.item()
                vn += 1
        val_loss = val_loss / max(vn, 1)

        scheduler.step(val_loss)

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch:3d}/{args.epochs} | "
              f"train_loss={train_loss:.6f} | "
              f"val_loss={val_loss:.6f} | "
              f"lr={lr:.2e} | {elapsed:.1f}s")

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            ckpt_path = checkpoint_dir / f"{name}_best.pt"
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                         "best_metric": best_loss}, ckpt_path)
            print(f"    -> New best! loss={best_loss:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n  Early stopping (best epoch: {best_epoch})")
                break

    # Load best
    ckpt_path = checkpoint_dir / f"{name}_best.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    # Test
    metrics = evaluate_intensity_reg_model(model, test_loader, str(device))
    print_intensity_reg_metrics(metrics, f"Test: {name}")
    return metrics


def train_intensity_cls_mlp(train_df, val_df, test_df, args, checkpoint_dir, name):
    """Train an IntensityClsMLP and evaluate on test set."""
    from src.data.intensity_dataset import IntensityClsEnvSingleDataset
    from src.models.intensity_models import IntensityClsMLP

    train_ds = IntensityClsEnvSingleDataset(train_df, DATA_ROOT)
    val_ds = IntensityClsEnvSingleDataset(val_df, DATA_ROOT)
    test_ds = IntensityClsEnvSingleDataset(test_df, DATA_ROOT)

    print(f"    Samples: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
        print(f"    Skipping {name}: insufficient data")
        return None

    train_labels = train_ds.df["future_inte_change24"].values.astype(np.int64)
    class_weights = compute_class_weights(train_labels, NUM_INTENSITY_CLASSES)
    sampler = make_weighted_sampler(train_labels, NUM_INTENSITY_CLASSES)

    bs = args.batch_size
    train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=bs * 2, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=bs * 2, shuffle=False,
                             num_workers=4, pin_memory=True)

    model = IntensityClsMLP()
    criterion = FocalLoss(gamma=FOCAL_GAMMA, weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    from src.training.trainer import Trainer
    trainer = Trainer(
        model=model, criterion=criterion, optimizer=optimizer,
        checkpoint_dir=checkpoint_dir, experiment_name=name,
    )
    trainer.train(train_loader, val_loader, epochs=args.epochs, patience=args.patience)

    metrics = evaluate_intensity_cls_model(model, test_loader, str(get_device()))
    print_intensity_cls_metrics(metrics, f"Test: {name}")
    return metrics


def train_intensity_reg_mlp(train_df, val_df, test_df, args, checkpoint_dir, name):
    """Train an IntensityRegMLP and evaluate on test set."""
    from src.data.intensity_dataset import IntensityRegEnvSingleDataset
    from src.models.intensity_models import IntensityRegMLP

    train_ds = IntensityRegEnvSingleDataset(train_df, DATA_ROOT)
    val_ds = IntensityRegEnvSingleDataset(val_df, DATA_ROOT)
    test_ds = IntensityRegEnvSingleDataset(test_df, DATA_ROOT)

    print(f"    Samples: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
        print(f"    Skipping {name}: insufficient data")
        return None

    bs = args.batch_size
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=bs * 2, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=bs * 2, shuffle=False,
                             num_workers=4, pin_memory=True)

    model = IntensityRegMLP()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    device = get_device()
    model = model.to(device)
    criterion = criterion.to(device)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    print(f"\n{'='*60}")
    print(f"  Training (Intensity Reg MLP): {name}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, n_batches = 0.0, 0
        for batch in train_loader:
            features, targets = batch
            features, targets = features.to(device), targets.to(device)
            preds = model(features)
            loss = criterion(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        train_loss = total_loss / max(n_batches, 1)

        model.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                features, targets = batch
                features, targets = features.to(device), targets.to(device)
                preds = model(features)
                vl += criterion(preds, targets).item()
                vn += 1
        val_loss = vl / max(vn, 1)
        scheduler.step(val_loss)

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch:3d}/{args.epochs} | "
              f"train={train_loss:.6f} | val={val_loss:.6f} | "
              f"lr={lr:.2e} | {elapsed:.1f}s")

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            ckpt_path = checkpoint_dir / f"{name}_best.pt"
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict()}, ckpt_path)
            print(f"    -> New best!")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n  Early stopping (best epoch: {best_epoch})")
                break

    ckpt_path = checkpoint_dir / f"{name}_best.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    metrics = evaluate_intensity_reg_model(model, test_loader, str(device))
    print_intensity_reg_metrics(metrics, f"Test: {name}")
    return metrics


# ── Temporal Experiment ───────────────────────────────────────────────────

MODEL_TRAINERS = {
    "cls_cnn": ("Intensity Cls CNN", train_intensity_cls_cnn),
    "reg_cnn": ("Intensity Reg CNN", train_intensity_reg_cnn),
    "cls_mlp": ("Intensity Cls MLP", train_intensity_cls_mlp),
    "reg_mlp": ("Intensity Reg MLP", train_intensity_reg_mlp),
}


def run_temporal_experiment(args, results_dir):
    """Train intensity models on different historical cutoffs."""
    print(f"\n{'#'*60}")
    print("  INTENSITY TEMPORAL GENERALIZATION")
    print(f"{'#'*60}")

    df = pd.read_csv(MASTER_INDEX_PATH)
    df = df[df["split"].isin(["train", "val", "test"])]

    # Test set is always 2017-2021
    test_df = df[df["year"] >= 2017]
    print(f"\n  Test set (2017-2021): {len(test_df):,} samples")

    cutoffs = [1990, 2000, 2010, 2016]
    results = {}

    for cutoff in cutoffs:
        print(f"\n{'='*60}")
        print(f"  Training period: 1950-{cutoff}")
        print(f"{'='*60}")

        period_df = df[(df["year"] <= cutoff) & (df["split"].isin(["train", "val"]))]

        if len(period_df) < 100:
            print(f"  Too few samples ({len(period_df)}), skipping")
            continue

        years = sorted(period_df["year"].unique())
        val_cutoff = years[int(len(years) * 0.8)]
        train_df = period_df[period_df["year"] <= val_cutoff]
        val_df = period_df[period_df["year"] > val_cutoff]

        print(f"  Train: {len(train_df):,} (1950-{val_cutoff})")
        print(f"  Val:   {len(val_df):,} ({val_cutoff+1}-{cutoff})")
        print(f"  Test:  {len(test_df):,} (2017-2021)")

        period_results = {}

        for model_key in args.models:
            if model_key not in MODEL_TRAINERS:
                print(f"  Unknown model: {model_key}, skipping")
                continue

            model_name, trainer_fn = MODEL_TRAINERS[model_key]
            exp_name = f"temporal_{model_key}_{cutoff}"

            print(f"\n  --- {model_name} (1950-{cutoff}) ---")

            metrics = trainer_fn(
                train_df, val_df, test_df, args,
                checkpoint_dir=results_dir / "checkpoints",
                name=exp_name,
            )

            if metrics:
                period_results[model_name] = metrics
                save_metrics(metrics, results_dir / f"{exp_name}.json")

        results[cutoff] = period_results

    # Build comparison table
    print(f"\n{'='*60}")
    print("  INTENSITY TEMPORAL GENERALIZATION RESULTS")
    print(f"{'='*60}")

    rows = []
    for cutoff in cutoffs:
        if cutoff not in results:
            continue
        for model_name, m in results[cutoff].items():
            row = {
                "Training Period": f"1950-{cutoff}",
                "Model": model_name,
            }
            # Add metrics depending on model type
            if "accuracy" in m:
                row["Accuracy"] = f"{m['accuracy']:.4f}"
                row["Macro F1"] = f"{m['macro_f1']:.4f}"
            if "mae_ms" in m:
                row["MAE (m/s)"] = f"{m['mae_ms']:.2f}"
                row["RMSE (m/s)"] = f"{m['rmse_ms']:.2f}"
                row["R2"] = f"{m['r2']:.4f}"
                row["Derived Cls Acc"] = f"{m.get('derived_cls_accuracy', 0):.4f}"
            rows.append(row)

    if rows:
        table_df = pd.DataFrame(rows)
        print(table_df.to_string(index=False))
        table_df.to_csv(results_dir / "intensity_temporal_comparison.csv", index=False)

    # Plot
    _plot_temporal_results(results, cutoffs, results_dir)

    return results


def _plot_temporal_results(results, cutoffs, results_dir):
    """Generate temporal generalization plots."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {
        "Intensity Cls CNN": "#FF6B6B",
        "Intensity Reg CNN": "#4ECDC4",
        "Intensity Cls MLP": "#FFE66D",
        "Intensity Reg MLP": "#A8E6CF",
    }
    markers = {
        "Intensity Cls CNN": "o",
        "Intensity Reg CNN": "s",
        "Intensity Cls MLP": "^",
        "Intensity Reg MLP": "D",
    }

    # Left panel: Classification accuracy or Regression MAE (m/s)
    all_model_names = set()
    for cutoff_results in results.values():
        all_model_names.update(cutoff_results.keys())

    # Classification models
    for model_name in sorted(all_model_names):
        if "Cls" not in model_name:
            continue
        cutoff_vals, acc_vals = [], []
        for cutoff in cutoffs:
            if cutoff in results and model_name in results[cutoff]:
                m = results[cutoff][model_name]
                cutoff_vals.append(cutoff)
                acc_vals.append(m.get("accuracy", 0))
        if cutoff_vals:
            axes[0].plot(cutoff_vals, acc_vals,
                         f"-{markers.get(model_name, 'o')}",
                         color=colors.get(model_name, "#888"),
                         label=model_name, linewidth=2, markersize=8)

    axes[0].set_xlabel("Training data cutoff year")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Intensity Classification Accuracy vs Training Period")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # Regression models
    for model_name in sorted(all_model_names):
        if "Reg" not in model_name:
            continue
        cutoff_vals, mae_vals = [], []
        for cutoff in cutoffs:
            if cutoff in results and model_name in results[cutoff]:
                m = results[cutoff][model_name]
                cutoff_vals.append(cutoff)
                mae_vals.append(m.get("mae_ms", 0))
        if cutoff_vals:
            axes[1].plot(cutoff_vals, mae_vals,
                         f"-{markers.get(model_name, 's')}",
                         color=colors.get(model_name, "#888"),
                         label=model_name, linewidth=2, markersize=8)

    axes[1].set_xlabel("Training data cutoff year")
    axes[1].set_ylabel("MAE (m/s)")
    axes[1].set_title("Intensity Regression MAE vs Training Period")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.suptitle(
        "Intensity Temporal Generalization:\n"
        "Do models trained on older data underpredict recent intensity changes?",
        fontweight="bold", fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(results_dir / "intensity_temporal_generalization.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved intensity_temporal_generalization.png")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Intensity temporal generalization experiment"
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--models", nargs="+", type=str,
                        default=["cls_cnn", "reg_cnn"],
                        help="Models to train. Options: cls_cnn, reg_cnn, cls_mlp, reg_mlp")
    args = parser.parse_args()

    set_seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = PROJECT_ROOT / "results" / f"intensity_temporal_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "checkpoints").mkdir(exist_ok=True)

    config = vars(args).copy()
    config["timestamp"] = timestamp
    config["device"] = str(get_device())
    with open(results_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n  Results directory: {results_dir}")
    print(f"  Device: {get_device()}")
    print(f"  Models: {args.models}")

    t_start = time.time()

    run_temporal_experiment(args, results_dir)

    total_time = time.time() - t_start
    print(f"\n{'#'*60}")
    print(f"  INTENSITY TEMPORAL EXPERIMENT COMPLETE")
    print(f"  Total time: {total_time/60:.1f} min")
    print(f"  Results: {results_dir}")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()
