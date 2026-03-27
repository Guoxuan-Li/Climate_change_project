"""
Run the full pipeline: all stages + baselines + ablation comparison.

Saves all results (metrics JSON, confusion matrices, training curves,
comparison table) to results/<timestamp>/.

Usage:
    python -m src.scripts.run_all
    python -m src.scripts.run_all --epochs 30 --stages 1 2 3
    python -m src.scripts.run_all --skip-stage4   # skip heavy CNN stage
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.config import (
    MASTER_INDEX_PATH, BATCH_SIZE, EPOCHS, LR, FOCAL_GAMMA, PATIENCE,
    NUM_DIRECTION_CLASSES, DATA_ROOT, DIRECTION_LABELS, SEED, PROJECT_ROOT,
)
from src.data.utils import compute_class_weights
from src.training.losses import FocalLoss
from src.training.trainer import Trainer, set_seed, get_device
from src.training.evaluate import (
    compute_metrics, print_metrics, evaluate_model, persistence_baseline,
    compute_regression_metrics, print_regression_metrics,
    evaluate_regression_model, _delta_to_direction_class,
)


# ── Helpers ────────────────────────────────────────────────────────────────

def load_index():
    df = pd.read_csv(MASTER_INDEX_PATH)
    df = df[df["future_direction24"] >= 0]
    df = df[df["split"].isin(["train", "val", "test"])]
    return df


def make_sampler(labels, num_classes):
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts = np.maximum(counts, 1.0)
    w = 1.0 / counts
    sample_w = w[labels]
    return WeightedRandomSampler(
        torch.tensor(sample_w, dtype=torch.float64),
        num_samples=len(labels), replacement=True,
    )


def save_confusion_matrix(cm, labels, path, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_training_curves(history, path, title="Training Curves"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["val_acc"])
    axes[1].set_title("Val Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(alpha=0.3)

    axes[2].plot(history["val_macro_f1"])
    axes[2].set_title("Val Macro F1")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(alpha=0.3)

    fig.suptitle(title, fontweight="bold")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_metrics(metrics, path):
    """Save metrics dict to JSON (converting numpy types)."""
    serializable = {}
    for k, v in metrics.items():
        if k == "confusion_matrix":
            serializable[k] = v.tolist()
        elif isinstance(v, np.floating):
            serializable[k] = float(v)
        elif isinstance(v, dict):
            serializable[k] = {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                                for kk, vv in v.items()}
        else:
            serializable[k] = v
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)


# ── Stage runners ──────────────────────────────────────────────────────────

def run_baselines(df, results_dir):
    """Compute majority-class and persistence baselines."""
    test_df = df[df["split"] == "test"]
    test_labels = test_df["future_direction24"].values

    # Majority class
    majority_class = int(pd.Series(df[df["split"] == "train"]["future_direction24"]).mode()[0])
    majority_preds = np.full_like(test_labels, majority_class)
    maj_metrics = compute_metrics(test_labels, majority_preds)
    print_metrics(maj_metrics, f"Majority Baseline (always predict {DIRECTION_LABELS[majority_class]})")
    save_metrics(maj_metrics, results_dir / "baseline_majority.json")

    # Persistence
    env_dicts = []
    for _, row in test_df.iterrows():
        d = np.load(str(DATA_ROOT / row["env_path"]), allow_pickle=True).item()
        env_dicts.append(d)
    y_true_p, y_pred_p = persistence_baseline(env_dicts)
    if len(y_true_p) > 0:
        pers_metrics = compute_metrics(y_true_p, y_pred_p)
        print_metrics(pers_metrics, "Persistence Baseline")
        save_metrics(pers_metrics, results_dir / "baseline_persistence.json")
        save_confusion_matrix(
            pers_metrics["confusion_matrix"], DIRECTION_LABELS,
            results_dir / "cm_persistence.png", "Persistence Baseline"
        )
        return {"Majority": maj_metrics, "Persistence": pers_metrics}

    return {"Majority": maj_metrics}


def run_stage1(df, args, results_dir):
    """Stage 1: Env-Data MLP."""
    from src.data.env_dataset import EnvSingleDataset
    from src.models.baseline_mlp import BaselineMLP

    train_df, val_df, test_df = [df[df["split"] == s] for s in ["train", "val", "test"]]

    train_ds = EnvSingleDataset(train_df, DATA_ROOT)
    val_ds = EnvSingleDataset(val_df, DATA_ROOT)
    test_ds = EnvSingleDataset(test_df, DATA_ROOT)

    train_labels = train_df["future_direction24"].values
    class_weights = compute_class_weights(train_labels, NUM_DIRECTION_CLASSES)
    sampler = make_sampler(train_labels, NUM_DIRECTION_CLASSES)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False,
                             num_workers=4, pin_memory=True)

    model = BaselineMLP()
    criterion = FocalLoss(gamma=args.focal_gamma, weight=class_weights)
    trainer = Trainer(model=model, criterion=criterion,
                      checkpoint_dir=results_dir / "checkpoints",
                      experiment_name="stage1_env_mlp")

    trainer.train(train_loader, val_loader, epochs=args.epochs, patience=args.patience)
    test_metrics = trainer.evaluate_on_test(test_loader)

    save_metrics(test_metrics, results_dir / "stage1_env_mlp.json")
    save_confusion_matrix(test_metrics["confusion_matrix"], DIRECTION_LABELS,
                          results_dir / "cm_stage1.png", "Stage 1: Env MLP")
    save_training_curves(trainer.history, results_dir / "curves_stage1.png", "Stage 1: Env MLP")

    return test_metrics


def run_stage2(df, args, results_dir):
    """Stage 2: LSTM on Data1D."""
    from src.data.data1d_dataset import Data1DSequenceDataset
    from src.models.lstm_1d import LSTMTracker

    train_df, val_df, test_df = [df[df["split"] == s] for s in ["train", "val", "test"]]

    train_ds = Data1DSequenceDataset(train_df, DATA_ROOT)
    val_ds = Data1DSequenceDataset(val_df, DATA_ROOT)
    test_ds = Data1DSequenceDataset(test_df, DATA_ROOT)

    train_labels = np.array([s["target"] for s in train_ds.samples])
    class_weights = compute_class_weights(train_labels, NUM_DIRECTION_CLASSES)
    sampler = make_sampler(train_labels, NUM_DIRECTION_CLASSES)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False,
                             num_workers=4, pin_memory=True)

    model = LSTMTracker()
    criterion = FocalLoss(gamma=args.focal_gamma, weight=class_weights)
    trainer = Trainer(model=model, criterion=criterion,
                      checkpoint_dir=results_dir / "checkpoints",
                      experiment_name="stage2_lstm_1d")

    trainer.train(train_loader, val_loader, epochs=args.epochs, patience=args.patience)
    test_metrics = trainer.evaluate_on_test(test_loader)

    save_metrics(test_metrics, results_dir / "stage2_lstm_1d.json")
    save_confusion_matrix(test_metrics["confusion_matrix"], DIRECTION_LABELS,
                          results_dir / "cm_stage2.png", "Stage 2: LSTM Data1D")
    save_training_curves(trainer.history, results_dir / "curves_stage2.png", "Stage 2: LSTM Data1D")

    return test_metrics


def run_stage3(df, args, results_dir):
    """Stage 3: Temporal Env-Data model."""
    from src.data.env_dataset import EnvSequenceDataset
    from src.models.env_temporal import EnvTemporalModel

    train_df, val_df, test_df = [df[df["split"] == s] for s in ["train", "val", "test"]]

    train_ds = EnvSequenceDataset(train_df, DATA_ROOT)
    val_ds = EnvSequenceDataset(val_df, DATA_ROOT)
    test_ds = EnvSequenceDataset(test_df, DATA_ROOT)

    train_labels = np.array([s["target"] for s in train_ds.samples])
    class_weights = compute_class_weights(train_labels, NUM_DIRECTION_CLASSES)
    sampler = make_sampler(train_labels, NUM_DIRECTION_CLASSES)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False,
                             num_workers=4, pin_memory=True)

    model = EnvTemporalModel()
    criterion = FocalLoss(gamma=args.focal_gamma, weight=class_weights)
    trainer = Trainer(model=model, criterion=criterion,
                      checkpoint_dir=results_dir / "checkpoints",
                      experiment_name="stage3_env_temporal")

    trainer.train(train_loader, val_loader, epochs=args.epochs, patience=args.patience)
    test_metrics = trainer.evaluate_on_test(test_loader)

    save_metrics(test_metrics, results_dir / "stage3_env_temporal.json")
    save_confusion_matrix(test_metrics["confusion_matrix"], DIRECTION_LABELS,
                          results_dir / "cm_stage3.png", "Stage 3: Env Temporal")
    save_training_curves(trainer.history, results_dir / "curves_stage3.png", "Stage 3: Env Temporal")

    return test_metrics


def run_stage4(df, args, results_dir):
    """Stage 4: CNN on Data3D."""
    from src.data.data3d_dataset import Data3DDataset
    from src.models.cnn_3d import CNNEncoder3D

    df_3d = df[df["data3d_exists"] == True]
    train_df, val_df, test_df = [df_3d[df_3d["split"] == s] for s in ["train", "val", "test"]]

    train_ds = Data3DDataset(train_df, DATA_ROOT)
    val_ds = Data3DDataset(val_df, DATA_ROOT)
    test_ds = Data3DDataset(test_df, DATA_ROOT)

    train_labels = train_df["future_direction24"].values
    class_weights = compute_class_weights(train_labels, NUM_DIRECTION_CLASSES)
    sampler = make_sampler(train_labels, NUM_DIRECTION_CLASSES)

    bs = max(args.batch_size // 2, 16)
    train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False,
                             num_workers=4, pin_memory=True)

    model = CNNEncoder3D()
    criterion = FocalLoss(gamma=args.focal_gamma, weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr * 0.1, weight_decay=1e-4)
    trainer = Trainer(model=model, criterion=criterion, optimizer=optimizer,
                      checkpoint_dir=results_dir / "checkpoints",
                      experiment_name="stage4_cnn_3d")

    trainer.train(train_loader, val_loader, epochs=args.epochs, patience=args.patience)
    test_metrics = trainer.evaluate_on_test(test_loader)

    save_metrics(test_metrics, results_dir / "stage4_cnn_3d.json")
    save_confusion_matrix(test_metrics["confusion_matrix"], DIRECTION_LABELS,
                          results_dir / "cm_stage4.png", "Stage 4: CNN Data3D")
    save_training_curves(trainer.history, results_dir / "curves_stage4.png", "Stage 4: CNN Data3D")

    return test_metrics


def run_stage5(df, args, results_dir):
    """Stage 5: Multimodal fusion."""
    from src.data.multimodal_dataset import MultimodalDataset
    from src.models.fusion_model import FusionModel

    df_3d = df[df["data3d_exists"] == True]
    train_df, val_df, test_df = [df_3d[df_3d["split"] == s] for s in ["train", "val", "test"]]

    print("  Building multimodal datasets (this may take a moment)...")
    train_ds = MultimodalDataset(train_df, DATA_ROOT)
    val_ds = MultimodalDataset(val_df, DATA_ROOT)
    test_ds = MultimodalDataset(test_df, DATA_ROOT)
    print(f"  Multimodal samples: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    train_labels = np.array([s["target"] for s in train_ds.samples])
    class_weights = compute_class_weights(train_labels, NUM_DIRECTION_CLASSES)
    sampler = make_sampler(train_labels, NUM_DIRECTION_CLASSES)

    bs = max(args.batch_size // 2, 16)
    train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False,
                             num_workers=4, pin_memory=True)

    model = FusionModel()
    criterion = FocalLoss(gamma=args.focal_gamma, weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr * 0.1, weight_decay=1e-4)
    trainer = Trainer(model=model, criterion=criterion, optimizer=optimizer,
                      checkpoint_dir=results_dir / "checkpoints",
                      experiment_name="stage5_fusion")

    trainer.train(train_loader, val_loader, epochs=args.epochs, patience=args.patience)
    test_metrics = trainer.evaluate_on_test(test_loader)

    save_metrics(test_metrics, results_dir / "stage5_fusion.json")
    save_confusion_matrix(test_metrics["confusion_matrix"], DIRECTION_LABELS,
                          results_dir / "cm_stage5.png", "Stage 5: Multimodal Fusion")
    save_training_curves(trainer.history, results_dir / "curves_stage5.png", "Stage 5: Fusion")

    return test_metrics


# ── Regression stage runners ───────────────────────────────────────────────

def run_persistence_regression(df, results_dir):
    """Persistence regression baseline: predict delta = same as last 24h displacement.

    For each test sample with valid regression target, look 4 timesteps back:
      pred_delta_lon = long_norm[t] - long_norm[t-4]
      pred_delta_lat = lat_norm[t] - lat_norm[t-4]
    (i.e., assume the cyclone will repeat its previous 24h movement.)
    """
    from src.config import DATA1D_ROOT, DATA1D_COLS, DATA1D_FEATURE_COLS

    test_df = df[(df["split"] == "test") &
                 df["delta_lon_norm"].notna() &
                 df["delta_lat_norm"].notna()].copy()
    test_df = test_df.sort_values(["year", "storm_name", "timestamp"]).reset_index(drop=True)

    y_true_list = []
    y_pred_list = []

    for (year, storm, d1d_file, split, basin), group in test_df.groupby(
        ["year", "storm_name", "data1d_file", "split", "basin"]
    ):
        d1d_path = DATA1D_ROOT / basin / split / d1d_file
        if not d1d_path.exists():
            continue

        track_df = pd.read_csv(
            d1d_path, delimiter='\t', header=None, names=DATA1D_COLS,
            dtype={"timestamp": str}
        )
        track_df["timestamp"] = track_df["timestamp"].astype(str).str.strip()
        ts_to_idx = {ts: i for i, ts in enumerate(track_df["timestamp"].values)}
        lon_arr = track_df["long_norm"].values.astype(float)
        lat_arr = track_df["lat_norm"].values.astype(float)

        for _, row in group.iterrows():
            ts = str(row["timestamp"]).strip()
            if ts not in ts_to_idx:
                continue
            idx = ts_to_idx[ts]
            # Need t-4 to exist for persistence
            if idx < 4:
                continue
            pred_dlon = lon_arr[idx] - lon_arr[idx - 4]
            pred_dlat = lat_arr[idx] - lat_arr[idx - 4]
            y_true_list.append([row["delta_lon_norm"], row["delta_lat_norm"]])
            y_pred_list.append([pred_dlon, pred_dlat])

    if len(y_true_list) == 0:
        print("  No valid persistence regression samples found.")
        return None

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)

    metrics = compute_regression_metrics(y_true, y_pred)
    print_regression_metrics(metrics, "Persistence Regression Baseline")
    save_metrics(metrics, results_dir / "reg_baseline_persistence.json")
    return metrics


def run_reg_stage1(df, args, results_dir):
    """Regression Stage 1: RegMLP on Env-Data."""
    from src.data.regression_dataset import RegEnvSingleDataset
    from src.models.regression_models import RegMLP
    from src.training.trainer import RegressionTrainer

    train_df, val_df, test_df = [df[df["split"] == s] for s in ["train", "val", "test"]]

    train_ds = RegEnvSingleDataset(train_df, DATA_ROOT)
    val_ds = RegEnvSingleDataset(val_df, DATA_ROOT)
    test_ds = RegEnvSingleDataset(test_df, DATA_ROOT)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False,
                             num_workers=4, pin_memory=True)

    model = RegMLP()
    criterion = torch.nn.MSELoss()
    trainer = RegressionTrainer(
        model=model, criterion=criterion,
        checkpoint_dir=results_dir / "checkpoints",
        experiment_name="reg_stage1_env_mlp",
    )

    trainer.train(train_loader, val_loader, epochs=args.epochs, patience=args.patience)
    test_metrics = trainer.evaluate_on_test(test_loader)
    save_metrics(test_metrics, results_dir / "reg_stage1_env_mlp.json")
    return test_metrics


def run_reg_stage2(df, args, results_dir):
    """Regression Stage 2: RegLSTM on Data1D."""
    from src.data.regression_dataset import RegData1DSequenceDataset
    from src.models.regression_models import RegLSTM
    from src.training.trainer import RegressionTrainer

    train_df, val_df, test_df = [df[df["split"] == s] for s in ["train", "val", "test"]]

    train_ds = RegData1DSequenceDataset(train_df, DATA_ROOT)
    val_ds = RegData1DSequenceDataset(val_df, DATA_ROOT)
    test_ds = RegData1DSequenceDataset(test_df, DATA_ROOT)

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
        checkpoint_dir=results_dir / "checkpoints",
        experiment_name="reg_stage2_lstm_1d",
    )

    trainer.train(train_loader, val_loader, epochs=args.epochs, patience=args.patience)
    test_metrics = trainer.evaluate_on_test(test_loader)
    save_metrics(test_metrics, results_dir / "reg_stage2_lstm_1d.json")
    return test_metrics


def run_reg_stage3(df, args, results_dir):
    """Regression Stage 3: RegEnvTemporal on Env-Data sequences."""
    from src.data.regression_dataset import RegEnvSequenceDataset
    from src.models.regression_models import RegEnvTemporal
    from src.training.trainer import RegressionTrainer

    train_df, val_df, test_df = [df[df["split"] == s] for s in ["train", "val", "test"]]

    train_ds = RegEnvSequenceDataset(train_df, DATA_ROOT)
    val_ds = RegEnvSequenceDataset(val_df, DATA_ROOT)
    test_ds = RegEnvSequenceDataset(test_df, DATA_ROOT)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False,
                             num_workers=4, pin_memory=True)

    model = RegEnvTemporal()
    criterion = torch.nn.MSELoss()
    trainer = RegressionTrainer(
        model=model, criterion=criterion,
        checkpoint_dir=results_dir / "checkpoints",
        experiment_name="reg_stage3_env_temporal",
    )

    trainer.train(train_loader, val_loader, epochs=args.epochs, patience=args.patience)
    test_metrics = trainer.evaluate_on_test(test_loader)
    save_metrics(test_metrics, results_dir / "reg_stage3_env_temporal.json")
    return test_metrics


def run_reg_stage4(df, args, results_dir):
    """Regression Stage 4: RegCNN3D on Data3D."""
    from src.data.regression_dataset import RegData3DDataset
    from src.models.regression_models import RegCNN3D
    from src.training.trainer import RegressionTrainer

    df_3d = df[df["data3d_exists"] == True]
    train_df, val_df, test_df = [df_3d[df_3d["split"] == s] for s in ["train", "val", "test"]]

    train_ds = RegData3DDataset(train_df, DATA_ROOT)
    val_ds = RegData3DDataset(val_df, DATA_ROOT)
    test_ds = RegData3DDataset(test_df, DATA_ROOT)

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
        checkpoint_dir=results_dir / "checkpoints",
        experiment_name="reg_stage4_cnn_3d",
    )

    trainer.train(train_loader, val_loader, epochs=args.epochs, patience=args.patience)
    test_metrics = trainer.evaluate_on_test(test_loader)
    save_metrics(test_metrics, results_dir / "reg_stage4_cnn_3d.json")
    return test_metrics


def run_reg_stage5(df, args, results_dir):
    """Regression Stage 5: RegFusionModel (multimodal)."""
    from src.data.regression_dataset import RegMultimodalDataset
    from src.models.regression_models import RegFusionModel
    from src.training.trainer import RegressionTrainer

    df_3d = df[df["data3d_exists"] == True]
    train_df, val_df, test_df = [df_3d[df_3d["split"] == s] for s in ["train", "val", "test"]]

    print("  Building regression multimodal datasets...")
    train_ds = RegMultimodalDataset(train_df, DATA_ROOT)
    val_ds = RegMultimodalDataset(val_df, DATA_ROOT)
    test_ds = RegMultimodalDataset(test_df, DATA_ROOT)
    print(f"  Reg multimodal samples: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    bs = max(args.batch_size // 2, 16)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False,
                             num_workers=4, pin_memory=True)

    model = RegFusionModel()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr * 0.1, weight_decay=1e-4)
    trainer = RegressionTrainer(
        model=model, criterion=criterion, optimizer=optimizer,
        checkpoint_dir=results_dir / "checkpoints",
        experiment_name="reg_stage5_fusion",
    )

    trainer.train(train_loader, val_loader, epochs=args.epochs, patience=args.patience)
    test_metrics = trainer.evaluate_on_test(test_loader)
    save_metrics(test_metrics, results_dir / "reg_stage5_fusion.json")
    return test_metrics


# ── Comparison table & summary ─────────────────────────────────────────────

def build_regression_comparison_table(reg_results: dict, results_dir: Path):
    """Build and save a comparison table for regression models."""
    rows = []
    for name, metrics in reg_results.items():
        rows.append({
            "Model": name,
            "MAE (km)": f"{metrics['mae_km']:.2f}",
            "Median Err (km)": f"{metrics['median_track_error_km']:.2f}",
            "Derived Acc": f"{metrics['derived_accuracy']:.4f}",
            "Derived F1": f"{metrics['derived_macro_f1']:.4f}",
            "R2 (dlon)": f"{metrics['r2_dlon']:.4f}",
            "R2 (dlat)": f"{metrics['r2_dlat']:.4f}",
        })

    table_df = pd.DataFrame(rows)
    table_df.to_csv(results_dir / "regression_comparison_table.csv", index=False)

    print(f"\n{'='*80}")
    print("  REGRESSION COMPARISON — ALL MODELS (Test Set)")
    print(f"{'='*80}")
    print(table_df.to_string(index=False))
    print()

    with open(results_dir / "regression_comparison_table.txt", "w") as f:
        f.write("REGRESSION COMPARISON — ALL MODELS (Test Set)\n")
        f.write("=" * 80 + "\n\n")
        f.write(table_df.to_string(index=False))
        f.write("\n")

    return table_df


def build_comparison_table(all_results: dict, results_dir: Path):
    """Build and save a comparison table across all models."""
    rows = []
    for name, metrics in all_results.items():
        rows.append({
            "Model": name,
            "Accuracy": f"{metrics['accuracy']:.4f}",
            "Macro F1": f"{metrics['macro_f1']:.4f}",
            "Weighted F1": f"{metrics['weighted_f1']:.4f}",
            **{f"F1({lbl})": f"{metrics['per_class_f1'].get(lbl, 0):.3f}"
               for lbl in DIRECTION_LABELS},
        })

    table_df = pd.DataFrame(rows)

    # Save as CSV
    table_df.to_csv(results_dir / "comparison_table.csv", index=False)

    # Print
    print(f"\n{'='*80}")
    print("  ABLATION COMPARISON — ALL MODELS (Test Set)")
    print(f"{'='*80}")
    print(table_df.to_string(index=False))
    print()

    # Save as formatted text
    with open(results_dir / "comparison_table.txt", "w") as f:
        f.write("ABLATION COMPARISON — ALL MODELS (Test Set)\n")
        f.write("=" * 80 + "\n\n")
        f.write(table_df.to_string(index=False))
        f.write("\n")

    # Bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    models = [r["Model"] for r in rows]
    accs = [float(r["Accuracy"]) for r in rows]
    f1s = [float(r["Macro F1"]) for r in rows]

    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

    axes[0].barh(models, accs, color=colors)
    axes[0].set_xlabel("Accuracy")
    axes[0].set_title("Test Accuracy")
    axes[0].set_xlim(0, 1)
    for i, v in enumerate(accs):
        axes[0].text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
    axes[0].grid(axis="x", alpha=0.3)

    axes[1].barh(models, f1s, color=colors)
    axes[1].set_xlabel("Macro F1")
    axes[1].set_title("Test Macro F1 (Primary Metric)")
    axes[1].set_xlim(0, 1)
    for i, v in enumerate(f1s):
        axes[1].text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
    axes[1].grid(axis="x", alpha=0.3)

    plt.suptitle("Model Comparison — Cyclone Direction Prediction (WP)", fontweight="bold")
    plt.tight_layout()
    fig.savefig(results_dir / "comparison_chart.png", dpi=150)
    plt.close(fig)

    return table_df


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run full cyclone prediction pipeline")
    parser.add_argument("--stages", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                        help="Which stages to run (default: all)")
    parser.add_argument("--skip-stage4", action="store_true",
                        help="Skip Stage 4 (CNN) — saves hours of compute")
    parser.add_argument("--skip-stage5", action="store_true",
                        help="Skip Stage 5 (Fusion)")
    parser.add_argument("--skip-regression", action="store_true",
                        help="Skip all regression stages")
    parser.add_argument("--regression-only", action="store_true",
                        help="Run only regression stages (skip classification)")
    parser.add_argument("--reg-stages", type=int, nargs="+", default=None,
                        help="Which regression stages to run (default: same as --stages)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--focal-gamma", type=float, default=FOCAL_GAMMA)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if args.skip_stage4 and 4 in args.stages:
        args.stages = [s for s in args.stages if s != 4]
    if args.skip_stage5 and 5 in args.stages:
        args.stages = [s for s in args.stages if s != 5]

    # Determine regression stages
    # Only inherit --skip-stage4/5 if --reg-stages was NOT explicitly provided
    if args.reg_stages is None:
        args.reg_stages = list(args.stages)  # inherits classification stage list + skips
    # If user explicitly set --reg-stages, respect it as-is (no skip filtering)

    set_seed(args.seed)

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = PROJECT_ROOT / "results" / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "checkpoints").mkdir(exist_ok=True)

    # Save run config
    config = vars(args).copy()
    config["timestamp"] = timestamp
    config["device"] = str(get_device())
    with open(results_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    run_cls = not args.regression_only
    run_reg = not args.skip_regression

    print(f"\n{'#'*60}")
    print(f"  CYCLONE TRAJECTORY PREDICTION — FULL PIPELINE")
    print(f"  Results directory: {results_dir}")
    print(f"  Classification stages: {args.stages if run_cls else 'SKIPPED'}")
    print(f"  Regression stages: {args.reg_stages if run_reg else 'SKIPPED'}")
    print(f"  Epochs: {args.epochs}, Batch size: {args.batch_size}")
    print(f"  Device: {get_device()}")
    print(f"{'#'*60}\n")

    # Load data
    df = load_index()
    print(f"  Total valid classification samples: {len(df):,}")
    for split in ["train", "val", "test"]:
        n = len(df[df["split"] == split])
        print(f"    {split}: {n:,}")

    # Regression sample count
    reg_df = df[df["delta_lon_norm"].notna() & df["delta_lat_norm"].notna()] \
        if "delta_lon_norm" in df.columns else pd.DataFrame()
    if not reg_df.empty:
        print(f"\n  Total valid regression samples: {len(reg_df):,}")
        for split in ["train", "val", "test"]:
            n = len(reg_df[reg_df["split"] == split])
            print(f"    {split}: {n:,}")
    print()

    all_results = {}
    reg_results = {}
    t_start = time.time()

    # ── Classification pipeline ────────────────────────────────────────────
    if run_cls:
        # Baselines
        print(f"\n{'='*60}")
        print("  CLASSIFICATION BASELINES")
        print(f"{'='*60}")
        baseline_results = run_baselines(df, results_dir)
        all_results.update(baseline_results)

        # Stages
        stage_runners = {
            1: ("Stage 1: Env MLP", run_stage1),
            2: ("Stage 2: LSTM Data1D", run_stage2),
            3: ("Stage 3: Env Temporal", run_stage3),
            4: ("Stage 4: CNN Data3D", run_stage4),
            5: ("Stage 5: Fusion", run_stage5),
        }

        for stage_num in sorted(args.stages):
            name, runner = stage_runners[stage_num]
            print(f"\n{'='*60}")
            print(f"  {name}")
            print(f"{'='*60}")
            t0 = time.time()
            try:
                metrics = runner(df, args, results_dir)
                elapsed = time.time() - t0
                all_results[name] = metrics
                print(f"\n  {name} completed in {elapsed/60:.1f} min")
            except Exception as e:
                print(f"\n  ERROR in {name}: {e}")
                import traceback
                traceback.print_exc()

        # Classification comparison
        build_comparison_table(all_results, results_dir)

    # ── Regression pipeline ────────────────────────────────────────────────
    if run_reg and not reg_df.empty:
        print(f"\n{'#'*60}")
        print("  REGRESSION PIPELINE")
        print(f"{'#'*60}")

        # Persistence regression baseline
        print(f"\n{'='*60}")
        print("  REGRESSION BASELINES")
        print(f"{'='*60}")
        pers_reg_metrics = run_persistence_regression(df, results_dir)
        if pers_reg_metrics is not None:
            reg_results["Persistence (Reg)"] = pers_reg_metrics

        reg_stage_runners = {
            1: ("Reg Stage 1: Env MLP", run_reg_stage1),
            2: ("Reg Stage 2: LSTM Data1D", run_reg_stage2),
            3: ("Reg Stage 3: Env Temporal", run_reg_stage3),
            4: ("Reg Stage 4: CNN Data3D", run_reg_stage4),
            5: ("Reg Stage 5: Fusion", run_reg_stage5),
        }

        for stage_num in sorted(args.reg_stages):
            name, runner = reg_stage_runners[stage_num]
            print(f"\n{'='*60}")
            print(f"  {name}")
            print(f"{'='*60}")
            t0 = time.time()
            try:
                metrics = runner(df, args, results_dir)
                elapsed = time.time() - t0
                reg_results[name] = metrics
                print(f"\n  {name} completed in {elapsed/60:.1f} min")
            except Exception as e:
                print(f"\n  ERROR in {name}: {e}")
                import traceback
                traceback.print_exc()

        # Regression comparison
        if reg_results:
            build_regression_comparison_table(reg_results, results_dir)

    elif run_reg and reg_df.empty:
        print("\n  WARNING: No regression targets found in index. "
              "Run build_index.py to add delta_lon_norm/delta_lat_norm columns.")

    # Trajectory visualizations
    try:
        from src.visualization.trajectory_plots import generate_test_visualizations
        generate_test_visualizations(
            results_dir=results_dir,
            checkpoint_dir=results_dir / "checkpoints",
            index_path=MASTER_INDEX_PATH,
            data_root=DATA_ROOT,
            num_storms=6,
        )
    except Exception as e:
        print(f"\n  WARNING: Trajectory visualization failed: {e}")
        import traceback
        traceback.print_exc()

    # Summary
    total_time = time.time() - t_start
    print(f"\n{'#'*60}")
    print(f"  PIPELINE COMPLETE")
    print(f"  Total time: {total_time/60:.1f} min")
    print(f"  Results saved to: {results_dir}")
    print(f"  Files:")
    for f in sorted(results_dir.rglob("*")):
        if f.is_file():
            size = f.stat().st_size
            print(f"    {f.relative_to(results_dir)} ({size/1024:.1f} KB)")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()
