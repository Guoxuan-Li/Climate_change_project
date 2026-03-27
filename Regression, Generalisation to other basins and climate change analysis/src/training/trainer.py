"""
Generic training loop with early stopping, LR scheduling, and logging.
"""
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import DEVICE, EPOCHS, LR, PATIENCE, WEIGHT_DECAY, SEED, PROJECT_ROOT
from src.training.evaluate import evaluate_model, print_metrics


def set_seed(seed: int = SEED):
    """Set all random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Get the best available device."""
    if DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class Trainer:
    """Generic trainer for classification models.

    Args:
        model: nn.Module
        criterion: loss function
        optimizer: torch optimizer (if None, creates Adam)
        scheduler: LR scheduler (if None, creates ReduceLROnPlateau)
        device: torch device
        checkpoint_dir: where to save best model
        experiment_name: name for logging
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer=None,
        scheduler=None,
        device=None,
        checkpoint_dir: str | Path | None = None,
        experiment_name: str = "experiment",
    ):
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)

        self.optimizer = optimizer or torch.optim.Adam(
            model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
        )
        self.scheduler = scheduler or torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=3
        )

        self.checkpoint_dir = Path(checkpoint_dir or PROJECT_ROOT / "checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name

        self.best_metric = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        self.history = {"train_loss": [], "val_loss": [], "val_macro_f1": [], "val_acc": []}

    def _train_one_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            if len(batch) == 2:
                features, targets = batch
            else:
                *features_list, targets = batch
                features = features_list

            if isinstance(features, (list, tuple)):
                features = [f.to(self.device) for f in features]
                logits = self.model(*features)
            else:
                features = features.to(self.device)
                logits = self.model(features)

            targets = targets.to(self.device)
            loss = self.criterion(logits, targets)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _val_loss(self, val_loader: DataLoader) -> float:
        """Compute validation loss."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in val_loader:
            if len(batch) == 2:
                features, targets = batch
            else:
                *features_list, targets = batch
                features = features_list

            if isinstance(features, (list, tuple)):
                features = [f.to(self.device) for f in features]
                logits = self.model(*features)
            else:
                features = features.to(self.device)
                logits = self.model(features)

            targets = targets.to(self.device)
            loss = self.criterion(logits, targets)
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = EPOCHS,
        patience: int = PATIENCE,
    ) -> dict:
        """Full training loop with early stopping on macro F1.

        Returns the best validation metrics.
        """
        set_seed()
        print(f"\n{'='*60}")
        print(f"  Training: {self.experiment_name}")
        print(f"  Device: {self.device}")
        print(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        print(f"{'='*60}\n")

        best_metrics = {}

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            # Train
            train_loss = self._train_one_epoch(train_loader)

            # Validate
            val_loss = self._val_loss(val_loader)
            val_metrics = evaluate_model(self.model, val_loader, str(self.device))

            val_f1 = val_metrics["macro_f1"]
            val_acc = val_metrics["accuracy"]

            # LR scheduling
            self.scheduler.step(val_f1)

            # Logging
            elapsed = time.time() - t0
            lr = self.optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"train_loss={train_loss:.4f} | "
                  f"val_loss={val_loss:.4f} | "
                  f"val_acc={val_acc:.4f} | "
                  f"val_macro_f1={val_f1:.4f} | "
                  f"lr={lr:.2e} | "
                  f"{elapsed:.1f}s")

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_macro_f1"].append(val_f1)
            self.history["val_acc"].append(val_acc)

            # Early stopping / checkpointing
            if val_f1 > self.best_metric:
                self.best_metric = val_f1
                self.best_epoch = epoch
                self.patience_counter = 0
                best_metrics = val_metrics

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
                          f"best macro_f1: {self.best_metric:.4f})")
                    break

        # Load best model
        ckpt_path = self.checkpoint_dir / f"{self.experiment_name}_best.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])
            print(f"\n  Loaded best model from epoch {ckpt['epoch']}")

        return best_metrics

    def evaluate_on_test(self, test_loader: DataLoader) -> dict:
        """Evaluate the (best) model on the test set."""
        metrics = evaluate_model(self.model, test_loader, str(self.device))
        print_metrics(metrics, f"Test Results: {self.experiment_name}")
        return metrics


class RegressionTrainer:
    """Trainer for regression models outputting (B, 2) continuous predictions.

    Uses MSE or similar loss. Early stopping based on validation MSE (lower is better).
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer=None,
        scheduler=None,
        device=None,
        checkpoint_dir: str | Path | None = None,
        experiment_name: str = "reg_experiment",
    ):
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

        self.best_metric = float("inf")  # track best val loss (lower = better)
        self.best_epoch = 0
        self.patience_counter = 0
        self.history = {
            "train_loss": [], "val_loss": [],
            "val_mae_km": [], "val_derived_acc": [],
        }

    def _run_batch(self, batch):
        """Unpack batch, run model, return (predictions, targets)."""
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

    def _train_one_epoch(self, train_loader: DataLoader) -> float:
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
    def _val_loss(self, val_loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in val_loader:
            preds, targets = self._run_batch(batch)
            loss = self.criterion(preds, targets)
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = EPOCHS,
        patience: int = PATIENCE,
    ) -> dict:
        """Full training loop with early stopping on val loss.

        Returns the best validation regression metrics.
        """
        from src.training.evaluate import evaluate_regression_model, print_regression_metrics

        set_seed()
        print(f"\n{'='*60}")
        print(f"  Training (Regression): {self.experiment_name}")
        print(f"  Device: {self.device}")
        print(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        print(f"{'='*60}\n")

        best_metrics = {}

        for epoch in range(1, epochs + 1):
            import time
            t0 = time.time()

            train_loss = self._train_one_epoch(train_loader)
            val_loss = self._val_loss(val_loader)

            # Full regression metrics on validation set
            val_metrics = evaluate_regression_model(
                self.model, val_loader, str(self.device)
            )
            val_mae_km = val_metrics["mae_km"]
            val_derived_acc = val_metrics["derived_accuracy"]

            self.scheduler.step(val_loss)

            elapsed = time.time() - t0
            lr = self.optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"train_loss={train_loss:.6f} | "
                  f"val_loss={val_loss:.6f} | "
                  f"val_mae_km={val_mae_km:.2f} | "
                  f"val_dir_acc={val_derived_acc:.4f} | "
                  f"lr={lr:.2e} | "
                  f"{elapsed:.1f}s")

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_mae_km"].append(val_mae_km)
            self.history["val_derived_acc"].append(val_derived_acc)

            # Early stopping on val_loss (lower is better)
            if val_loss < self.best_metric:
                self.best_metric = val_loss
                self.best_epoch = epoch
                self.patience_counter = 0
                best_metrics = val_metrics

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

        return best_metrics

    def evaluate_on_test(self, test_loader: DataLoader) -> dict:
        """Evaluate the (best) regression model on the test set."""
        from src.training.evaluate import evaluate_regression_model, print_regression_metrics
        metrics = evaluate_regression_model(
            self.model, test_loader, str(self.device)
        )
        print_regression_metrics(metrics, f"Test Results: {self.experiment_name}")
        return metrics
