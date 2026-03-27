"""
CLI training entry point for all model stages.

Usage:
    python -m src.scripts.train --stage 1              # Env-Data MLP baseline
    python -m src.scripts.train --stage 2              # LSTM on Data1D
    python -m src.scripts.train --stage 3              # Env-Data temporal
    python -m src.scripts.train --stage 4              # CNN on Data3D
    python -m src.scripts.train --stage 5              # Multimodal fusion
    python -m src.scripts.train --stage 1 --epochs 30 --batch-size 128
"""
import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.config import (
    MASTER_INDEX_PATH, BATCH_SIZE, EPOCHS, LR, FOCAL_GAMMA, PATIENCE,
    NUM_DIRECTION_CLASSES, ENV_FEATURE_DIM, HIDDEN_DIM, SEED, DATA_ROOT,
    SEQ_LEN, DATA1D_NUM_FEATURES,
)
from src.data.utils import compute_class_weights
from src.training.losses import FocalLoss
from src.training.trainer import Trainer, set_seed, get_device
from src.training.evaluate import print_metrics, compute_metrics
from src.training.evaluate import (
    print_regression_metrics, evaluate_regression_model, compute_regression_metrics,
)


def load_index(path=MASTER_INDEX_PATH):
    df = pd.read_csv(path)
    # Filter valid labels and known splits
    df = df[df["future_direction24"] >= 0]
    df = df[df["split"].isin(["train", "val", "test"])]
    return df


def load_index_regression(path=MASTER_INDEX_PATH):
    """Load index filtered for valid regression targets."""
    df = pd.read_csv(path)
    df = df[df["split"].isin(["train", "val", "test"])]
    df = df[df["delta_lon_norm"].notna() & df["delta_lat_norm"].notna()]
    return df


def make_weighted_sampler(labels: np.ndarray, num_classes: int) -> WeightedRandomSampler:
    """Create a WeightedRandomSampler for balanced training."""
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts = np.maximum(counts, 1.0)
    weight_per_class = 1.0 / counts
    sample_weights = weight_per_class[labels]
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float64),
        num_samples=len(labels),
        replacement=True,
    )


def train_stage1(args):
    """Stage 1: MLP on single-timestep Env-Data."""
    from src.data.env_dataset import EnvSingleDataset
    from src.models.baseline_mlp import BaselineMLP

    df = load_index()
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    train_ds = EnvSingleDataset(train_df, data_root=DATA_ROOT)
    val_ds = EnvSingleDataset(val_df, data_root=DATA_ROOT)
    test_ds = EnvSingleDataset(test_df, data_root=DATA_ROOT)

    # Class weights and sampler
    train_labels = train_df["future_direction24"].values
    class_weights = compute_class_weights(train_labels, NUM_DIRECTION_CLASSES)
    sampler = make_weighted_sampler(train_labels, NUM_DIRECTION_CLASSES)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False,
                             num_workers=4, pin_memory=True)

    model = BaselineMLP()
    criterion = FocalLoss(gamma=args.focal_gamma, weight=class_weights)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        experiment_name="stage1_env_mlp",
    )

    # Train
    best_val = trainer.train(train_loader, val_loader, epochs=args.epochs, patience=args.patience)
    print_metrics(best_val, "Best Validation — Stage 1 Env MLP")

    # Test
    test_metrics = trainer.evaluate_on_test(test_loader)

    # Persistence baseline on test set for comparison
    print("\n  Computing persistence baseline on test set...")
    from src.training.evaluate import persistence_baseline
    env_dicts = []
    for _, row in test_df.iterrows():
        env_path = DATA_ROOT / row["env_path"]
        d = np.load(str(env_path), allow_pickle=True).item()
        env_dicts.append(d)
    y_true_p, y_pred_p = persistence_baseline(env_dicts)
    if len(y_true_p) > 0:
        p_metrics = compute_metrics(y_true_p, y_pred_p)
        print_metrics(p_metrics, "Persistence Baseline (test set, valid history only)")

    return test_metrics


def train_stage2(args):
    """Stage 2: LSTM on Data1D sequences."""
    from src.data.data1d_dataset import Data1DSequenceDataset
    from src.models.lstm_1d import LSTMTracker

    df = load_index()
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    train_ds = Data1DSequenceDataset(train_df, data_root=DATA_ROOT)
    val_ds = Data1DSequenceDataset(val_df, data_root=DATA_ROOT)
    test_ds = Data1DSequenceDataset(test_df, data_root=DATA_ROOT)

    # Class weights
    train_labels = np.array([s["target"] for s in train_ds.samples])
    class_weights = compute_class_weights(train_labels, NUM_DIRECTION_CLASSES)
    sampler = make_weighted_sampler(train_labels, NUM_DIRECTION_CLASSES)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False,
                             num_workers=4, pin_memory=True)

    model = LSTMTracker()
    criterion = FocalLoss(gamma=args.focal_gamma, weight=class_weights)

    trainer = Trainer(model=model, criterion=criterion, experiment_name="stage2_lstm_1d")
    best_val = trainer.train(train_loader, val_loader, epochs=args.epochs, patience=args.patience)
    print_metrics(best_val, "Best Validation — Stage 2 LSTM 1D")
    trainer.evaluate_on_test(test_loader)


def train_stage3(args):
    """Stage 3: Temporal model on Env-Data sequences."""
    from src.data.env_dataset import EnvSequenceDataset
    from src.models.env_temporal import EnvTemporalModel

    df = load_index()
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    train_ds = EnvSequenceDataset(train_df, data_root=DATA_ROOT)
    val_ds = EnvSequenceDataset(val_df, data_root=DATA_ROOT)
    test_ds = EnvSequenceDataset(test_df, data_root=DATA_ROOT)

    train_labels = np.array([s["target"] for s in train_ds.samples])
    class_weights = compute_class_weights(train_labels, NUM_DIRECTION_CLASSES)
    sampler = make_weighted_sampler(train_labels, NUM_DIRECTION_CLASSES)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False,
                             num_workers=4, pin_memory=True)

    model = EnvTemporalModel()
    criterion = FocalLoss(gamma=args.focal_gamma, weight=class_weights)

    trainer = Trainer(model=model, criterion=criterion, experiment_name="stage3_env_temporal")
    best_val = trainer.train(train_loader, val_loader, epochs=args.epochs, patience=args.patience)
    print_metrics(best_val, "Best Validation — Stage 3 Env Temporal")
    trainer.evaluate_on_test(test_loader)


def train_stage4(args):
    """Stage 4: CNN on Data3D fields."""
    from src.data.data3d_dataset import Data3DDataset
    from src.models.cnn_3d import CNNEncoder3D

    df = load_index()
    # Only use samples where Data3D exists
    df = df[df["data3d_exists"] == True]
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    train_ds = Data3DDataset(train_df, data_root=DATA_ROOT)
    val_ds = Data3DDataset(val_df, data_root=DATA_ROOT)
    test_ds = Data3DDataset(test_df, data_root=DATA_ROOT)

    train_labels = train_df["future_direction24"].values
    class_weights = compute_class_weights(train_labels, NUM_DIRECTION_CLASSES)
    sampler = make_weighted_sampler(train_labels, NUM_DIRECTION_CLASSES)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size // 2, sampler=sampler,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    model = CNNEncoder3D()
    criterion = FocalLoss(gamma=args.focal_gamma, weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr * 0.1, weight_decay=1e-4)
    trainer = Trainer(model=model, criterion=criterion, optimizer=optimizer,
                      experiment_name="stage4_cnn_3d")
    best_val = trainer.train(train_loader, val_loader, epochs=args.epochs, patience=args.patience)
    print_metrics(best_val, "Best Validation — Stage 4 CNN 3D")
    trainer.evaluate_on_test(test_loader)


def train_stage5(args):
    """Stage 5: Multimodal fusion."""
    from src.data.multimodal_dataset import MultimodalDataset
    from src.models.fusion_model import FusionModel

    df = load_index()
    df = df[df["data3d_exists"] == True]
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    train_ds = MultimodalDataset(train_df, data_root=DATA_ROOT)
    val_ds = MultimodalDataset(val_df, data_root=DATA_ROOT)
    test_ds = MultimodalDataset(test_df, data_root=DATA_ROOT)

    train_labels = train_df["future_direction24"].values
    class_weights = compute_class_weights(train_labels, NUM_DIRECTION_CLASSES)
    sampler = make_weighted_sampler(train_labels, NUM_DIRECTION_CLASSES)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size // 2, sampler=sampler,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    model = FusionModel()
    criterion = FocalLoss(gamma=args.focal_gamma, weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr * 0.1, weight_decay=1e-4)
    trainer = Trainer(model=model, criterion=criterion, optimizer=optimizer,
                      experiment_name="stage5_fusion")
    best_val = trainer.train(train_loader, val_loader, epochs=args.epochs, patience=args.patience)
    print_metrics(best_val, "Best Validation — Stage 5 Fusion")
    trainer.evaluate_on_test(test_loader)


def train_reg_stage1(args):
    """Regression Stage 1: RegMLP on Env-Data."""
    from src.data.regression_dataset import RegEnvSingleDataset
    from src.models.regression_models import RegMLP

    df = load_index_regression()
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    train_ds = RegEnvSingleDataset(train_df, data_root=DATA_ROOT)
    val_ds = RegEnvSingleDataset(val_df, data_root=DATA_ROOT)
    test_ds = RegEnvSingleDataset(test_df, data_root=DATA_ROOT)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False,
                             num_workers=4, pin_memory=True)

    model = RegMLP()
    criterion = torch.nn.MSELoss()

    from src.training.trainer import RegressionTrainer
    trainer = RegressionTrainer(
        model=model, criterion=criterion,
        experiment_name="reg_stage1_env_mlp",
    )

    trainer.train(train_loader, val_loader, epochs=args.epochs, patience=args.patience)
    test_metrics = trainer.evaluate_on_test(test_loader)
    return test_metrics


def train_reg_stage2(args):
    """Regression Stage 2: RegLSTM on Data1D."""
    from src.data.regression_dataset import RegData1DSequenceDataset
    from src.models.regression_models import RegLSTM

    df = load_index_regression()
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    train_ds = RegData1DSequenceDataset(train_df, data_root=DATA_ROOT)
    val_ds = RegData1DSequenceDataset(val_df, data_root=DATA_ROOT)
    test_ds = RegData1DSequenceDataset(test_df, data_root=DATA_ROOT)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False,
                             num_workers=4, pin_memory=True)

    model = RegLSTM()
    criterion = torch.nn.MSELoss()

    from src.training.trainer import RegressionTrainer
    trainer = RegressionTrainer(
        model=model, criterion=criterion,
        experiment_name="reg_stage2_lstm_1d",
    )

    trainer.train(train_loader, val_loader, epochs=args.epochs, patience=args.patience)
    test_metrics = trainer.evaluate_on_test(test_loader)
    return test_metrics


def train_reg_stage3(args):
    """Regression Stage 3: RegEnvTemporal on Env-Data sequences."""
    from src.data.regression_dataset import RegEnvSequenceDataset
    from src.models.regression_models import RegEnvTemporal

    df = load_index_regression()
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    train_ds = RegEnvSequenceDataset(train_df, data_root=DATA_ROOT)
    val_ds = RegEnvSequenceDataset(val_df, data_root=DATA_ROOT)
    test_ds = RegEnvSequenceDataset(test_df, data_root=DATA_ROOT)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False,
                             num_workers=4, pin_memory=True)

    model = RegEnvTemporal()
    criterion = torch.nn.MSELoss()

    from src.training.trainer import RegressionTrainer
    trainer = RegressionTrainer(
        model=model, criterion=criterion,
        experiment_name="reg_stage3_env_temporal",
    )

    trainer.train(train_loader, val_loader, epochs=args.epochs, patience=args.patience)
    test_metrics = trainer.evaluate_on_test(test_loader)
    return test_metrics


def train_reg_stage4(args):
    """Regression Stage 4: RegCNN3D on Data3D."""
    from src.data.regression_dataset import RegData3DDataset
    from src.models.regression_models import RegCNN3D

    df = load_index_regression()
    df = df[df["data3d_exists"] == True]
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    train_ds = RegData3DDataset(train_df, data_root=DATA_ROOT)
    val_ds = RegData3DDataset(val_df, data_root=DATA_ROOT)
    test_ds = RegData3DDataset(test_df, data_root=DATA_ROOT)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size // 2, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    model = RegCNN3D()
    criterion = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr * 0.1, weight_decay=1e-4)
    from src.training.trainer import RegressionTrainer
    trainer = RegressionTrainer(
        model=model, criterion=criterion, optimizer=optimizer,
        experiment_name="reg_stage4_cnn_3d",
    )

    trainer.train(train_loader, val_loader, epochs=args.epochs, patience=args.patience)
    test_metrics = trainer.evaluate_on_test(test_loader)
    return test_metrics


def train_reg_stage5(args):
    """Regression Stage 5: RegFusionModel (multimodal)."""
    from src.data.regression_dataset import RegMultimodalDataset
    from src.models.regression_models import RegFusionModel

    df = load_index_regression()
    df = df[df["data3d_exists"] == True]
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    train_ds = RegMultimodalDataset(train_df, data_root=DATA_ROOT)
    val_ds = RegMultimodalDataset(val_df, data_root=DATA_ROOT)
    test_ds = RegMultimodalDataset(test_df, data_root=DATA_ROOT)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size // 2, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    model = RegFusionModel()
    criterion = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr * 0.1, weight_decay=1e-4)
    from src.training.trainer import RegressionTrainer
    trainer = RegressionTrainer(
        model=model, criterion=criterion, optimizer=optimizer,
        experiment_name="reg_stage5_fusion",
    )

    trainer.train(train_loader, val_loader, epochs=args.epochs, patience=args.patience)
    test_metrics = trainer.evaluate_on_test(test_loader)
    return test_metrics


def main():
    parser = argparse.ArgumentParser(description="Train cyclone trajectory models")
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2, 3, 4, 5],
                        help="Model stage to train (1-5)")
    parser.add_argument("--mode", type=str, default="classification",
                        choices=["classification", "regression"],
                        help="Training mode: classification (default) or regression")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--focal-gamma", type=float, default=FOCAL_GAMMA)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    set_seed(args.seed)

    if args.mode == "classification":
        stage_fn = {1: train_stage1, 2: train_stage2, 3: train_stage3,
                    4: train_stage4, 5: train_stage5}
    else:
        stage_fn = {1: train_reg_stage1, 2: train_reg_stage2, 3: train_reg_stage3,
                    4: train_reg_stage4, 5: train_reg_stage5}

    stage_fn[args.stage](args)


if __name__ == "__main__":
    main()
