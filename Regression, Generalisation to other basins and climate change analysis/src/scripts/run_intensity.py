"""
Intensity prediction pipeline: train classification and regression models.

Trains 4 classification models (MLP, LSTM, Env Temporal, CNN) and
4 regression models (same architectures) for 24h intensity change prediction.

Also computes persistence baseline and produces comparison tables.

Usage:
    python -m src.scripts.run_intensity --epochs 50
    python -m src.scripts.run_intensity --epochs 50 --cls-only
    python -m src.scripts.run_intensity --epochs 50 --reg-only
    python -m src.scripts.run_intensity --epochs 50 --stages 1 4
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
from src.training.trainer import Trainer, RegressionTrainer, set_seed, get_device
from src.training.evaluate import (
    compute_intensity_cls_metrics, print_intensity_cls_metrics,
    evaluate_intensity_cls_model,
    compute_intensity_reg_metrics, print_intensity_reg_metrics,
    evaluate_intensity_reg_model,
)


# ── Helpers ────────────────────────────────────────────────────────────────

def save_metrics(metrics, path):
    """Save metrics dict to JSON, handling numpy types."""
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


def load_index():
    """Load master index, filtered to valid splits."""
    df = pd.read_csv(MASTER_INDEX_PATH)
    df = df[df["split"].isin(["train", "val", "test"])]
    return df


def print_class_distribution(labels, label_names, title="Class Distribution"):
    """Print class distribution for intensity labels."""
    print(f"\n  {title}:")
    counts = np.bincount(labels, minlength=len(label_names))
    total = len(labels)
    for i, name in enumerate(label_names):
        pct = counts[i] / total * 100 if total > 0 else 0
        print(f"    {i} ({name:>14s}): {counts[i]:>6,} ({pct:5.1f}%)")
    return counts


def make_weighted_sampler(labels: np.ndarray, num_classes: int) -> WeightedRandomSampler:
    """Create a WeightedRandomSampler for class-balanced sampling."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weight_per_class = 1.0 / counts
    sample_weights = weight_per_class[labels]
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(labels),
        replacement=True,
    )


# ── Persistence Baseline ──────────────────────────────────────────────────

def intensity_persistence_baseline(index_df: pd.DataFrame) -> dict:
    """Persistence baseline: predict future_inte_change24 = history_inte_change24.

    Only uses samples where both are valid.
    """
    valid_df = index_df[
        (index_df["future_inte_change24"] >= 0) &
        (index_df["has_history24"] == True)
    ].copy()

    y_true = []
    y_pred = []

    for _, row in valid_df.iterrows():
        env_path = DATA_ROOT / row["env_path"]
        try:
            env_dict = np.load(str(env_path), allow_pickle=True).item()
        except Exception:
            continue

        future = env_dict.get("future_inte_change24", -1)
        hist = env_dict.get("history_inte_change24", -1)

        if isinstance(future, (int, float)) and future == -1:
            continue
        if isinstance(hist, (int, float)) and hist == -1:
            continue

        # hist is a one-hot array
        if hasattr(hist, "__len__"):
            y_pred.append(int(np.argmax(hist)))
        else:
            continue

        if hasattr(future, "item"):
            y_true.append(int(future.item()))
        else:
            y_true.append(int(future))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print(f"\n  Intensity Persistence Baseline: {len(y_true):,} samples with valid history+future")

    if len(y_true) == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "note": "No valid samples"}

    metrics = compute_intensity_cls_metrics(y_true, y_pred)
    print_intensity_cls_metrics(metrics, "Intensity Persistence Baseline")
    return metrics


# ── Classification Training ──────────────────────────────────────────────

def train_intensity_cls(stage, train_df, val_df, test_df, args, results_dir):
    """Train a single intensity classification model."""

    stage_config = {
        1: {
            "name": "int_cls_mlp",
            "title": "Intensity Cls MLP (Env Single)",
            "model_cls": "IntensityClsMLP",
            "dataset_cls": "IntensityClsEnvSingleDataset",
            "dataset_type": "env_single",
            "lr_factor": 1.0,
            "bs_factor": 1,
        },
        2: {
            "name": "int_cls_lstm",
            "title": "Intensity Cls LSTM (Data1D Seq)",
            "model_cls": "IntensityClsLSTM",
            "dataset_cls": "IntensityRegData1DSequenceDataset",  # reuse for sliding windows
            "dataset_type": "data1d_seq",
            "lr_factor": 1.0,
            "bs_factor": 1,
        },
        3: {
            "name": "int_cls_env_temporal",
            "title": "Intensity Cls Env Temporal (Transformer)",
            "model_cls": "IntensityClsEnvTemporal",
            "dataset_cls": "IntensityClsEnvSequenceDataset",
            "dataset_type": "env_seq",
            "lr_factor": 1.0,
            "bs_factor": 1,
        },
        4: {
            "name": "int_cls_cnn",
            "title": "Intensity Cls CNN (Data3D)",
            "model_cls": "IntensityClsCNN",
            "dataset_cls": "IntensityClsData3DDataset",
            "dataset_type": "data3d",
            "lr_factor": 0.1,
            "bs_factor": 0.5,
        },
    }

    cfg = stage_config[stage]
    print(f"\n{'#'*60}")
    print(f"  {cfg['title']}")
    print(f"{'#'*60}")

    # Import dataset and model classes
    from src.data.intensity_dataset import (
        IntensityClsEnvSingleDataset, IntensityClsEnvSequenceDataset,
        IntensityClsData3DDataset,
    )
    from src.data.data1d_dataset import Data1DSequenceDataset
    from src.models.intensity_models import (
        IntensityClsMLP, IntensityClsLSTM, IntensityClsEnvTemporal, IntensityClsCNN,
    )

    # Build datasets
    if cfg["dataset_type"] == "env_single":
        train_ds = IntensityClsEnvSingleDataset(train_df, DATA_ROOT)
        val_ds = IntensityClsEnvSingleDataset(val_df, DATA_ROOT)
        test_ds = IntensityClsEnvSingleDataset(test_df, DATA_ROOT)
    elif cfg["dataset_type"] == "data1d_seq":
        # For LSTM intensity cls, we need Data1D windows with intensity cls target
        # We'll build a custom approach: use Data1DSequenceDataset pattern but with intensity target
        train_ds = _build_data1d_intensity_cls_dataset(train_df)
        val_ds = _build_data1d_intensity_cls_dataset(val_df)
        test_ds = _build_data1d_intensity_cls_dataset(test_df)
    elif cfg["dataset_type"] == "env_seq":
        train_ds = IntensityClsEnvSequenceDataset(train_df, DATA_ROOT)
        val_ds = IntensityClsEnvSequenceDataset(val_df, DATA_ROOT)
        test_ds = IntensityClsEnvSequenceDataset(test_df, DATA_ROOT)
    elif cfg["dataset_type"] == "data3d":
        train_ds = IntensityClsData3DDataset(train_df, DATA_ROOT)
        val_ds = IntensityClsData3DDataset(val_df, DATA_ROOT)
        test_ds = IntensityClsData3DDataset(test_df, DATA_ROOT)
    else:
        raise ValueError(f"Unknown dataset type: {cfg['dataset_type']}")

    print(f"  Samples: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    if len(train_ds) == 0 or len(val_ds) == 0:
        print(f"  Skipping {cfg['name']}: insufficient data")
        return None

    # Get labels for class weighting
    train_labels = _extract_labels(train_ds)
    counts = print_class_distribution(train_labels, INTENSITY_LABELS,
                                       f"Train set class distribution ({cfg['name']})")

    # Class weights and weighted sampler
    class_weights = compute_class_weights(train_labels, NUM_INTENSITY_CLASSES)
    sampler = make_weighted_sampler(train_labels, NUM_INTENSITY_CLASSES)

    bs = max(int(args.batch_size * cfg["bs_factor"]), 16)
    train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=bs * 2, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=bs * 2, shuffle=False,
                             num_workers=4, pin_memory=True)

    # Build model
    model_map = {
        "IntensityClsMLP": IntensityClsMLP,
        "IntensityClsLSTM": IntensityClsLSTM,
        "IntensityClsEnvTemporal": IntensityClsEnvTemporal,
        "IntensityClsCNN": IntensityClsCNN,
    }
    model = model_map[cfg["model_cls"]]()

    # Loss: FocalLoss with class weights
    criterion = FocalLoss(gamma=FOCAL_GAMMA, weight=class_weights)

    # Optimizer
    lr = args.lr * cfg["lr_factor"]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    checkpoint_dir = results_dir / "checkpoints"
    trainer = Trainer(
        model=model, criterion=criterion, optimizer=optimizer,
        checkpoint_dir=checkpoint_dir,
        experiment_name=cfg["name"],
    )

    # Override the evaluate_model call to use intensity metrics
    # We use the standard Trainer which calls evaluate_model (8-class),
    # but we need 4-class. Since Trainer early-stops on macro_f1 from
    # evaluate_model and those functions work generically on any num_classes
    # produced by argmax, this actually works fine — the Trainer just calls
    # evaluate_model which does accuracy_score/f1_score on argmax predictions.
    # The num_classes=8 default in compute_metrics doesn't affect accuracy/f1
    # computation, only the confusion matrix and per-class stats. For training
    # loop purposes this is fine.

    trainer.train(train_loader, val_loader, epochs=args.epochs, patience=args.patience)

    # Proper evaluation with 4-class metrics
    metrics = evaluate_intensity_cls_model(model, test_loader, str(get_device()))
    print_intensity_cls_metrics(metrics, f"Test: {cfg['title']}")

    save_metrics(metrics, results_dir / f"{cfg['name']}_test.json")

    return metrics


class _Data1DIntensityClsDataset(torch.utils.data.Dataset):
    """Data1D sliding-window dataset with intensity classification target."""

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (torch.tensor(s["features"], dtype=torch.float32),
                torch.tensor(s["target"], dtype=torch.long))


def _build_data1d_intensity_cls_dataset(index_df, seq_len=8):
    """Build a Data1D sliding-window dataset for intensity classification."""
    from src.config import DATA1D_COLS, DATA1D_FEATURE_COLS, DATA1D_ROOT, SEQ_LEN

    valid_df = index_df[index_df["future_inte_change24"] >= 0].copy()
    valid_df = valid_df.sort_values(
        ["year", "storm_name", "timestamp"]
    ).reset_index(drop=True)

    # Build lookup
    ts_to_label = {}
    for _, row in valid_df.iterrows():
        key = (row["year"], row["storm_name"], str(row["timestamp"]))
        ts_to_label[key] = int(row["future_inte_change24"])

    samples = []
    storm_groups = valid_df.groupby(
        ["year", "storm_name", "data1d_file", "split", "basin"]
    )

    for (year, storm_name, d1d_file, split, basin), group in storm_groups:
        d1d_path = DATA1D_ROOT / basin / split / d1d_file
        if not d1d_path.exists():
            continue

        track_df = pd.read_csv(
            d1d_path, delimiter='\t', header=None, names=DATA1D_COLS,
            dtype={"timestamp": str}
        )
        track_df["timestamp"] = track_df["timestamp"].astype(str).str.strip()

        features = track_df[DATA1D_FEATURE_COLS].values.astype(np.float32)
        timestamps = track_df["timestamp"].values

        for i in range(len(track_df) - seq_len + 1):
            last_ts = timestamps[i + seq_len - 1]
            key = (year, storm_name, last_ts)
            if key in ts_to_label:
                samples.append({
                    "features": features[i:i + seq_len].copy(),
                    "target": ts_to_label[key],
                })

    return _Data1DIntensityClsDataset(samples)


def _extract_labels(dataset) -> np.ndarray:
    """Extract all target labels from a dataset."""
    if hasattr(dataset, "df"):
        # EnvSingle, Data3D — have a DataFrame
        if "future_inte_change24" in dataset.df.columns:
            return dataset.df["future_inte_change24"].values.astype(np.int64)
    if hasattr(dataset, "samples"):
        # Sequence datasets, custom datasets
        return np.array([s["target"] for s in dataset.samples], dtype=np.int64)
    # Fallback: iterate (slow)
    labels = []
    for i in range(len(dataset)):
        _, target = dataset[i]
        labels.append(target.item() if hasattr(target, "item") else int(target))
    return np.array(labels, dtype=np.int64)


# ── Regression Training ──────────────────────────────────────────────────

def train_intensity_reg(stage, train_df, val_df, test_df, args, results_dir):
    """Train a single intensity regression model."""

    stage_config = {
        1: {
            "name": "int_reg_mlp",
            "title": "Intensity Reg MLP (Env Single)",
            "model_cls": "IntensityRegMLP",
            "dataset_type": "env_single",
            "lr_factor": 1.0,
            "bs_factor": 1,
        },
        2: {
            "name": "int_reg_lstm",
            "title": "Intensity Reg LSTM (Data1D Seq)",
            "model_cls": "IntensityRegLSTM",
            "dataset_type": "data1d_seq",
            "lr_factor": 1.0,
            "bs_factor": 1,
        },
        3: {
            "name": "int_reg_env_temporal",
            "title": "Intensity Reg Env Temporal (Transformer)",
            "model_cls": "IntensityRegEnvTemporal",
            "dataset_type": "env_seq",
            "lr_factor": 1.0,
            "bs_factor": 1,
        },
        4: {
            "name": "int_reg_cnn",
            "title": "Intensity Reg CNN (Data3D)",
            "model_cls": "IntensityRegCNN",
            "dataset_type": "data3d",
            "lr_factor": 0.1,
            "bs_factor": 0.5,
        },
    }

    cfg = stage_config[stage]
    print(f"\n{'#'*60}")
    print(f"  {cfg['title']}")
    print(f"{'#'*60}")

    from src.data.intensity_dataset import (
        IntensityRegEnvSingleDataset, IntensityRegData1DSequenceDataset,
        IntensityRegData3DDataset,
    )
    from src.models.intensity_models import (
        IntensityRegMLP, IntensityRegLSTM, IntensityRegEnvTemporal, IntensityRegCNN,
    )

    # Build datasets
    if cfg["dataset_type"] == "env_single":
        train_ds = IntensityRegEnvSingleDataset(train_df, DATA_ROOT)
        val_ds = IntensityRegEnvSingleDataset(val_df, DATA_ROOT)
        test_ds = IntensityRegEnvSingleDataset(test_df, DATA_ROOT)
    elif cfg["dataset_type"] == "data1d_seq":
        train_ds = IntensityRegData1DSequenceDataset(train_df, DATA_ROOT)
        val_ds = IntensityRegData1DSequenceDataset(val_df, DATA_ROOT)
        test_ds = IntensityRegData1DSequenceDataset(test_df, DATA_ROOT)
    elif cfg["dataset_type"] == "env_seq":
        # Build env sequence dataset for intensity regression
        train_ds = _build_env_seq_intensity_reg_dataset(train_df)
        val_ds = _build_env_seq_intensity_reg_dataset(val_df)
        test_ds = _build_env_seq_intensity_reg_dataset(test_df)
    elif cfg["dataset_type"] == "data3d":
        train_ds = IntensityRegData3DDataset(train_df, DATA_ROOT)
        val_ds = IntensityRegData3DDataset(val_df, DATA_ROOT)
        test_ds = IntensityRegData3DDataset(test_df, DATA_ROOT)
    else:
        raise ValueError(f"Unknown dataset type: {cfg['dataset_type']}")

    print(f"  Samples: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    if len(train_ds) == 0 or len(val_ds) == 0:
        print(f"  Skipping {cfg['name']}: insufficient data")
        return None

    bs = max(int(args.batch_size * cfg["bs_factor"]), 16)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=bs * 2, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=bs * 2, shuffle=False,
                             num_workers=4, pin_memory=True)

    # Build model
    model_map = {
        "IntensityRegMLP": IntensityRegMLP,
        "IntensityRegLSTM": IntensityRegLSTM,
        "IntensityRegEnvTemporal": IntensityRegEnvTemporal,
        "IntensityRegCNN": IntensityRegCNN,
    }
    model = model_map[cfg["model_cls"]]()

    criterion = torch.nn.MSELoss()

    lr = args.lr * cfg["lr_factor"]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    checkpoint_dir = results_dir / "checkpoints"

    # Use a custom IntensityRegressionTrainer since RegressionTrainer
    # calls evaluate_regression_model which expects (B,2) output.
    # We need (B,1) output with intensity-specific metrics.
    trainer = _IntensityRegTrainer(
        model=model, criterion=criterion, optimizer=optimizer,
        checkpoint_dir=checkpoint_dir,
        experiment_name=cfg["name"],
    )

    trainer.train(train_loader, val_loader, epochs=args.epochs, patience=args.patience)

    # Proper evaluation with intensity regression metrics
    metrics = evaluate_intensity_reg_model(model, test_loader, str(get_device()))
    print_intensity_reg_metrics(metrics, f"Test: {cfg['title']}")

    save_metrics(metrics, results_dir / f"{cfg['name']}_test.json")

    return metrics


class _EnvSeqIntensityRegDataset(torch.utils.data.Dataset):
    """Env-Data sequence dataset for intensity regression."""

    def __init__(self, samples, data_root):
        self.samples = samples
        self.data_root = data_root

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from src.data.utils import env_dict_to_vector
        sample = self.samples[idx]
        seq = []
        for env_rel_path in sample["env_paths"]:
            env_path = self.data_root / env_rel_path
            env_dict = np.load(str(env_path), allow_pickle=True).item()
            seq.append(env_dict_to_vector(env_dict))

        features = np.stack(seq, axis=0)
        target = sample["target"]

        return (torch.tensor(features, dtype=torch.float32),
                torch.tensor(target, dtype=torch.float32).unsqueeze(0))


def _build_env_seq_intensity_reg_dataset(index_df, seq_len=8):
    """Build Env sequence dataset for intensity regression."""
    from src.config import SEQ_LEN

    valid_df = index_df[index_df["delta_wnd_norm"].notna()].copy()
    valid_df = valid_df.sort_values(
        ["storm_name", "year", "timestamp"]
    ).reset_index(drop=True)

    samples = []
    for (year, storm), group in valid_df.groupby(["year", "storm_name"]):
        group = group.sort_values("timestamp").reset_index(drop=True)
        if len(group) < seq_len:
            continue
        for i in range(len(group) - seq_len + 1):
            window = group.iloc[i:i + seq_len]
            last_row = window.iloc[-1]
            samples.append({
                "env_paths": window["env_path"].tolist(),
                "target": np.float32(last_row["delta_wnd_norm"]),
            })

    return _EnvSeqIntensityRegDataset(samples, DATA_ROOT)


class _IntensityRegTrainer:
    """Trainer for intensity regression models outputting (B, 1).

    Early stops on validation MSE (lower is better).
    """

    def __init__(self, model, criterion, optimizer=None, scheduler=None,
                 device=None, checkpoint_dir=None, experiment_name="int_reg"):
        from src.config import DEVICE, LR, WEIGHT_DECAY, PROJECT_ROOT
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)

        self.optimizer = optimizer or torch.optim.Adam(
            model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
        )
        self.scheduler = scheduler or torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3
        )

        self.checkpoint_dir = Path(checkpoint_dir or PROJECT_ROOT / "checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name

        self.best_metric = float("inf")
        self.best_epoch = 0
        self.patience_counter = 0
        self.history = {"train_loss": [], "val_loss": []}

    def _run_batch(self, batch):
        if len(batch) == 2:
            features, targets = batch
        else:
            *features_list, targets = batch
            features = features_list

        if isinstance(features, (list, tuple)):
            features = [f.to(self.device) for f in features]
            preds = self.model(*features)
        else:
            features = features.to(self.device)
            preds = self.model(features)

        targets = targets.to(self.device)
        return preds, targets

    def _train_one_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            preds, targets = self._run_batch(batch)
            loss = self.criterion(preds, targets)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _val_loss(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in val_loader:
            preds, targets = self._run_batch(batch)
            loss = self.criterion(preds, targets)
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def train(self, train_loader, val_loader, epochs=EPOCHS, patience=PATIENCE):
        set_seed()
        print(f"\n{'='*60}")
        print(f"  Training (Intensity Reg): {self.experiment_name}")
        print(f"  Device: {self.device}")
        print(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        print(f"{'='*60}\n")

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            train_loss = self._train_one_epoch(train_loader)
            val_loss = self._val_loss(val_loader)

            self.scheduler.step(val_loss)

            elapsed = time.time() - t0
            lr = self.optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"train_loss={train_loss:.6f} | "
                  f"val_loss={val_loss:.6f} | "
                  f"lr={lr:.2e} | "
                  f"{elapsed:.1f}s")

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            if val_loss < self.best_metric:
                self.best_metric = val_loss
                self.best_epoch = epoch
                self.patience_counter = 0

                ckpt_path = self.checkpoint_dir / f"{self.experiment_name}_best.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "best_metric": self.best_metric,
                }, ckpt_path)
                print(f"    -> New best! Saved to {ckpt_path.name}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    print(f"\n  Early stopping at epoch {epoch} "
                          f"(best epoch: {self.best_epoch}, "
                          f"best val_loss: {self.best_metric:.6f})")
                    break

        # Load best model
        ckpt_path = self.checkpoint_dir / f"{self.experiment_name}_best.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])
            print(f"\n  Loaded best model from epoch {ckpt['epoch']}")


# ── Comparison Tables ─────────────────────────────────────────────────────

def build_cls_comparison_table(all_metrics, results_dir):
    """Build and save classification comparison table."""
    rows = []
    for name, m in all_metrics.items():
        if m is None:
            continue
        row = {
            "Model": name,
            "Accuracy": f"{m['accuracy']:.4f}",
            "Macro F1": f"{m['macro_f1']:.4f}",
            "Weighted F1": f"{m.get('weighted_f1', 0):.4f}",
        }
        # Per-class F1
        for label in INTENSITY_LABELS:
            if "per_class_f1" in m and label in m["per_class_f1"]:
                row[f"F1 {label}"] = f"{m['per_class_f1'][label]:.3f}"
        rows.append(row)

    if rows:
        table = pd.DataFrame(rows)
        print(f"\n{'='*60}")
        print("  INTENSITY CLASSIFICATION COMPARISON")
        print(f"{'='*60}")
        print(table.to_string(index=False))
        table.to_csv(results_dir / "intensity_cls_comparison.csv", index=False)


def build_reg_comparison_table(all_metrics, results_dir):
    """Build and save regression comparison table."""
    rows = []
    for name, m in all_metrics.items():
        if m is None:
            continue
        rows.append({
            "Model": name,
            "MAE (norm)": f"{m['mae_norm']:.4f}",
            "RMSE (norm)": f"{m['rmse_norm']:.4f}",
            "MAE (m/s)": f"{m['mae_ms']:.2f}",
            "RMSE (m/s)": f"{m['rmse_ms']:.2f}",
            "R2": f"{m['r2']:.4f}",
            "Derived Cls Acc": f"{m['derived_cls_accuracy']:.4f}",
            "Derived Cls F1": f"{m['derived_cls_macro_f1']:.4f}",
        })

    if rows:
        table = pd.DataFrame(rows)
        print(f"\n{'='*60}")
        print("  INTENSITY REGRESSION COMPARISON")
        print(f"{'='*60}")
        print(table.to_string(index=False))
        table.to_csv(results_dir / "intensity_reg_comparison.csv", index=False)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Intensity prediction pipeline")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--stages", nargs="+", type=int, default=[1, 2, 3, 4],
                        help="Which stages to train (1=MLP, 2=LSTM, 3=Transformer, 4=CNN)")
    parser.add_argument("--cls-only", action="store_true",
                        help="Only train classification models")
    parser.add_argument("--reg-only", action="store_true",
                        help="Only train regression models")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip persistence baseline computation")
    args = parser.parse_args()

    set_seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = PROJECT_ROOT / "results" / f"intensity_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "checkpoints").mkdir(exist_ok=True)

    # Save config
    config = vars(args).copy()
    config["timestamp"] = timestamp
    config["device"] = str(get_device())
    with open(results_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n  Results directory: {results_dir}")
    print(f"  Device: {get_device()}")

    t_start = time.time()

    # Load index
    df = load_index()
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    print(f"\n  Index loaded: train={len(train_df):,}, val={len(val_df):,}, test={len(test_df):,}")

    # Check intensity class distribution
    valid_int = df[df["future_inte_change24"] >= 0]
    print(f"\n  Intensity classification samples (future_inte_change24 >= 0):")
    print(f"    Total valid: {len(valid_int):,}")
    print_class_distribution(
        valid_int["future_inte_change24"].values.astype(np.int64),
        INTENSITY_LABELS, "Overall intensity class distribution"
    )

    # Check intensity regression stats
    valid_wnd = df[df["delta_wnd_norm"].notna()]
    if len(valid_wnd) > 0:
        print(f"\n  Intensity regression samples (delta_wnd_norm valid): {len(valid_wnd):,}")
        print(f"    mean={valid_wnd['delta_wnd_norm'].mean():.4f}, "
              f"std={valid_wnd['delta_wnd_norm'].std():.4f}")

    # ── Persistence Baseline ──
    cls_metrics = {}
    reg_metrics = {}

    if not args.skip_baseline:
        print(f"\n{'#'*60}")
        print("  COMPUTING PERSISTENCE BASELINE")
        print(f"{'#'*60}")
        baseline = intensity_persistence_baseline(test_df)
        cls_metrics["Persistence"] = baseline
        save_metrics(baseline, results_dir / "baseline_persistence.json")

    # ── Classification ──
    run_cls = not args.reg_only
    run_reg = not args.cls_only

    if run_cls:
        print(f"\n{'#'*60}")
        print("  INTENSITY CLASSIFICATION MODELS")
        print(f"{'#'*60}")

        stage_names = {1: "MLP", 2: "LSTM", 3: "Env Temporal", 4: "CNN"}

        for stage in args.stages:
            m = train_intensity_cls(stage, train_df, val_df, test_df, args, results_dir)
            if m is not None:
                cls_metrics[f"Cls {stage}: {stage_names[stage]}"] = m

        build_cls_comparison_table(cls_metrics, results_dir)

    # ── Regression ──
    if run_reg:
        print(f"\n{'#'*60}")
        print("  INTENSITY REGRESSION MODELS")
        print(f"{'#'*60}")

        stage_names = {1: "MLP", 2: "LSTM", 3: "Env Temporal", 4: "CNN"}

        for stage in args.stages:
            m = train_intensity_reg(stage, train_df, val_df, test_df, args, results_dir)
            if m is not None:
                reg_metrics[f"Reg {stage}: {stage_names[stage]}"] = m

        build_reg_comparison_table(reg_metrics, results_dir)

    # ── Summary ──
    total_time = time.time() - t_start
    print(f"\n{'#'*60}")
    print(f"  INTENSITY PIPELINE COMPLETE")
    print(f"  Total time: {total_time/60:.1f} min")
    print(f"  Results: {results_dir}")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()
