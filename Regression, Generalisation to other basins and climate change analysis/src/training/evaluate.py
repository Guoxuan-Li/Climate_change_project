"""
Evaluation utilities: metrics computation, confusion matrices, reporting.

Supports both classification and regression evaluation.
"""
import math

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, r2_score,
)

from src.config import (
    DIRECTION_LABELS, NUM_DIRECTION_CLASSES,
    INTENSITY_LABELS, NUM_INTENSITY_CLASSES,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    num_classes: int = NUM_DIRECTION_CLASSES,
                    labels: list[str] = DIRECTION_LABELS) -> dict:
    """Compute all evaluation metrics for direction classification.

    Returns dict with: accuracy, macro_f1, weighted_f1, per_class_f1,
    per_class_precision, per_class_recall, confusion_matrix, report_str.
    """
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    report = classification_report(
        y_true, y_pred, labels=list(range(num_classes)),
        target_names=labels, zero_division=0
    )

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_f1": dict(zip(labels, f1.tolist())),
        "per_class_precision": dict(zip(labels, prec.tolist())),
        "per_class_recall": dict(zip(labels, rec.tolist())),
        "per_class_support": dict(zip(labels, support.tolist())),
        "confusion_matrix": cm,
        "report_str": report,
    }


def print_metrics(metrics: dict, title: str = "Evaluation Results"):
    """Pretty-print evaluation metrics."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Macro F1:    {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"\n{metrics['report_str']}")


@torch.no_grad()
def evaluate_model(model, dataloader, device: str = "cuda") -> dict:
    """Run model on a dataloader and compute all metrics.

    Args:
        model: nn.Module in eval mode
        dataloader: yields (features, targets) batches
        device: target device

    Returns:
        metrics dict from compute_metrics
    """
    model.eval()
    all_preds = []
    all_targets = []

    for batch in dataloader:
        if len(batch) == 2:
            features, targets = batch
        else:
            *features_list, targets = batch
            features = features_list

        if isinstance(features, (list, tuple)):
            features = [f.to(device) for f in features]
            logits = model(*features)
        else:
            features = features.to(device)
            logits = model(features)

        targets = targets.to(device)
        preds = logits.argmax(dim=1)

        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)

    return compute_metrics(y_true, y_pred)


def _delta_to_direction_class(dlon: np.ndarray, dlat: np.ndarray) -> np.ndarray:
    """Convert (dlon, dlat) displacement to 8-class direction bin.

    Uses atan2 to get angle, then maps to nearest of the 8 cardinal directions.
    Convention: 0=E, 1=NE, 2=N, 3=NW, 4=W, 5=SW, 6=S, 7=SE
    """
    angles = np.arctan2(dlat, dlon)  # radians, range [-pi, pi]
    # Shift to [0, 2*pi)
    angles = angles % (2 * np.pi)
    # Each bin is 45 degrees = pi/4. Bin centres at 0, pi/4, pi/2, ...
    # Shift by half a bin (pi/8) so that bin boundaries fall between centres
    bin_idx = np.floor((angles + np.pi / 8) / (np.pi / 4)).astype(int) % 8
    return bin_idx


def _haversine_km(dlon_deg: np.ndarray, dlat_deg: np.ndarray,
                  ref_lat_deg: float = 20.0) -> np.ndarray:
    """Compute great-circle distance in km from degree displacements.

    Uses a reference latitude for the cos(lat) longitude correction.
    """
    R = 6371.0
    dlon_rad = np.radians(dlon_deg)
    dlat_rad = np.radians(dlat_deg)
    ref_lat_rad = np.radians(ref_lat_deg)

    a = (np.sin(dlat_rad / 2) ** 2 +
         np.cos(ref_lat_rad) * np.cos(ref_lat_rad) *
         np.sin(dlon_rad / 2) ** 2)
    a = np.clip(a, 0.0, 1.0)
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                               ref_lat_deg: float = 20.0) -> dict:
    """Compute regression evaluation metrics for displacement prediction.

    Args:
        y_true: (N, 2) array of (delta_lon_norm, delta_lat_norm) ground truth.
        y_pred: (N, 2) array of (delta_lon_norm, delta_lat_norm) predictions.
        ref_lat_deg: Reference latitude for Haversine approximation.

    Returns:
        dict with: mae_km, mean_track_error_km, median_track_error_km,
        derived_accuracy, derived_macro_f1, r2_dlon, r2_dlat, mse_norm.
    """
    NORM_TO_DEG = 5.0  # delta_norm * 5 = delta degrees

    # Convert to degree deltas
    true_dlon_deg = y_true[:, 0] * NORM_TO_DEG
    true_dlat_deg = y_true[:, 1] * NORM_TO_DEG
    pred_dlon_deg = y_pred[:, 0] * NORM_TO_DEG
    pred_dlat_deg = y_pred[:, 1] * NORM_TO_DEG

    # Track errors in km (Haversine distance between predicted and actual displacement)
    error_dlon_deg = pred_dlon_deg - true_dlon_deg
    error_dlat_deg = pred_dlat_deg - true_dlat_deg
    track_errors_km = _haversine_km(error_dlon_deg, error_dlat_deg, ref_lat_deg)

    mae_km = np.mean(track_errors_km)
    mean_track_error_km = np.mean(track_errors_km)
    median_track_error_km = np.median(track_errors_km)

    # R-squared for each component (in normalized space)
    # Clamp predictions to avoid overflow in r2_score / MSE
    y_pred_clamped = np.clip(y_pred, -50.0, 50.0)

    try:
        r2_dlon = float(r2_score(y_true[:, 0], y_pred_clamped[:, 0]))
    except Exception:
        r2_dlon = -999.0
    try:
        r2_dlat = float(r2_score(y_true[:, 1], y_pred_clamped[:, 1]))
    except Exception:
        r2_dlat = -999.0

    # Replace -inf/inf/nan with safe values for JSON serialization
    if not np.isfinite(r2_dlon):
        r2_dlon = -999.0
    if not np.isfinite(r2_dlat):
        r2_dlat = -999.0

    # MSE in normalized space
    mse_norm = float(np.mean((y_true - y_pred_clamped) ** 2))
    if not np.isfinite(mse_norm):
        mse_norm = 999999.0

    # Derived classification metrics: convert both to direction bins
    true_dirs = _delta_to_direction_class(y_true[:, 0], y_true[:, 1])
    pred_dirs = _delta_to_direction_class(y_pred[:, 0], y_pred[:, 1])

    derived_accuracy = float(accuracy_score(true_dirs, pred_dirs))
    derived_macro_f1 = float(f1_score(true_dirs, pred_dirs, average="macro",
                                       zero_division=0))

    return {
        "mae_km": float(mae_km),
        "mean_track_error_km": float(mean_track_error_km),
        "median_track_error_km": float(median_track_error_km),
        "derived_accuracy": derived_accuracy,
        "derived_macro_f1": derived_macro_f1,
        "r2_dlon": r2_dlon,
        "r2_dlat": r2_dlat,
        "mse_norm": mse_norm,
    }


def print_regression_metrics(metrics: dict, title: str = "Regression Results"):
    """Pretty-print regression evaluation metrics."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  MAE (km):             {metrics['mae_km']:.2f}")
    print(f"  Mean Track Error (km): {metrics['mean_track_error_km']:.2f}")
    print(f"  Median Track Err (km): {metrics['median_track_error_km']:.2f}")
    print(f"  R2 (dlon):            {metrics['r2_dlon']:.4f}")
    print(f"  R2 (dlat):            {metrics['r2_dlat']:.4f}")
    print(f"  MSE (norm):           {metrics['mse_norm']:.6f}")
    print(f"  Derived Accuracy:     {metrics['derived_accuracy']:.4f}")
    print(f"  Derived Macro F1:     {metrics['derived_macro_f1']:.4f}")


@torch.no_grad()
def evaluate_regression_model(model, dataloader, device: str = "cuda") -> dict:
    """Run a regression model on a dataloader and compute regression metrics.

    Args:
        model: nn.Module outputting (B, 2) predictions
        dataloader: yields batches ending with (B, 2) float targets
        device: target device

    Returns:
        metrics dict from compute_regression_metrics
    """
    model.eval()
    all_preds = []
    all_targets = []

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

        targets = targets.to(device)

        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)

    return compute_regression_metrics(y_true, y_pred)


def _delta_wnd_to_intensity_class(delta_wnd_norm: np.ndarray) -> np.ndarray:
    """Map predicted delta_wnd_norm to 4-class intensity change bins.

    Reproduces the same binning logic as the dataset authors' env_data.py:
      0 = Strengthening  (delta_wnd > threshold, consistently increasing)
      1 = Str-then-Weak  (mixed: increased then decreased)
      2 = Weakening      (delta_wnd < -threshold, consistently decreasing)
      3 = Maintaining    (small change)

    Since we only have a single 24h delta (not the per-6h breakdown), we
    use a simplified binning:
      delta > 0.08  -> 0 (Strengthening)   [~2 m/s increase]
      -0.04 < delta <= 0.08 -> 3 (Maintaining)
      delta <= -0.04 -> 2 (Weakening)
    Class 1 (Str-then-Weak) cannot be determined from a single scalar;
    we never predict it from regression output.

    Note: thresholds were chosen to approximately match the actual class
    distribution in the WP dataset.
    """
    classes = np.full(len(delta_wnd_norm), 3, dtype=np.int64)  # default: maintaining
    classes[delta_wnd_norm > 0.08] = 0     # strengthening
    classes[delta_wnd_norm <= -0.04] = 2   # weakening
    return classes


def compute_intensity_cls_metrics(
    y_true: np.ndarray, y_pred: np.ndarray,
    num_classes: int = NUM_INTENSITY_CLASSES,
    labels: list[str] = INTENSITY_LABELS,
) -> dict:
    """Compute evaluation metrics for intensity classification (4 classes).

    Returns dict with: accuracy, macro_f1, weighted_f1, per_class_f1,
    per_class_precision, per_class_recall, confusion_matrix, report_str.
    """
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    report = classification_report(
        y_true, y_pred, labels=list(range(num_classes)),
        target_names=labels, zero_division=0
    )

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class_f1": dict(zip(labels, f1.tolist())),
        "per_class_precision": dict(zip(labels, prec.tolist())),
        "per_class_recall": dict(zip(labels, rec.tolist())),
        "per_class_support": dict(zip(labels, support.tolist())),
        "confusion_matrix": cm,
        "report_str": report,
    }


def print_intensity_cls_metrics(metrics: dict, title: str = "Intensity Cls Results"):
    """Pretty-print intensity classification metrics."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Macro F1:    {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"\n{metrics['report_str']}")


def compute_intensity_reg_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute evaluation metrics for intensity regression.

    Args:
        y_true: (N,) or (N,1) array of ground-truth delta_wnd_norm.
        y_pred: (N,) or (N,1) array of predicted delta_wnd_norm.

    Returns:
        dict with: mae_norm, rmse_norm, mae_ms, rmse_ms, r2,
        derived_cls_accuracy, derived_cls_macro_f1.
    """
    WND_SCALE = 25.0  # wnd_norm = (wnd-40)/25, so delta_ms = delta_norm * 25

    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    # Clamp predictions to avoid overflow
    y_pred_c = np.clip(y_pred, -10.0, 10.0)

    # Errors in normalized space
    errors = y_pred_c - y_true
    mae_norm = float(np.mean(np.abs(errors)))
    rmse_norm = float(np.sqrt(np.mean(errors ** 2)))

    # Physical units (m/s)
    mae_ms = mae_norm * WND_SCALE
    rmse_ms = rmse_norm * WND_SCALE

    # R-squared
    try:
        r2 = float(r2_score(y_true, y_pred_c))
    except Exception:
        r2 = -999.0
    if not np.isfinite(r2):
        r2 = -999.0

    # MSE in normalized space
    mse_norm = float(np.mean(errors ** 2))
    if not np.isfinite(mse_norm):
        mse_norm = 999999.0

    # Derived 4-class accuracy: bin both into intensity classes and compare
    true_classes = _delta_wnd_to_intensity_class(y_true)
    pred_classes = _delta_wnd_to_intensity_class(y_pred_c)

    derived_cls_accuracy = float(accuracy_score(true_classes, pred_classes))
    derived_cls_macro_f1 = float(f1_score(true_classes, pred_classes,
                                           average="macro", zero_division=0))

    return {
        "mae_norm": mae_norm,
        "rmse_norm": rmse_norm,
        "mae_ms": mae_ms,
        "rmse_ms": rmse_ms,
        "r2": r2,
        "mse_norm": mse_norm,
        "derived_cls_accuracy": derived_cls_accuracy,
        "derived_cls_macro_f1": derived_cls_macro_f1,
    }


def print_intensity_reg_metrics(metrics: dict, title: str = "Intensity Reg Results"):
    """Pretty-print intensity regression metrics."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  MAE (norm):           {metrics['mae_norm']:.4f}")
    print(f"  RMSE (norm):          {metrics['rmse_norm']:.4f}")
    print(f"  MAE (m/s):            {metrics['mae_ms']:.2f}")
    print(f"  RMSE (m/s):           {metrics['rmse_ms']:.2f}")
    print(f"  R2:                   {metrics['r2']:.4f}")
    print(f"  Derived Cls Acc:      {metrics['derived_cls_accuracy']:.4f}")
    print(f"  Derived Cls Macro F1: {metrics['derived_cls_macro_f1']:.4f}")


@torch.no_grad()
def evaluate_intensity_cls_model(model, dataloader, device: str = "cuda") -> dict:
    """Run an intensity classification model on a dataloader and compute metrics."""
    model.eval()
    all_preds = []
    all_targets = []

    for batch in dataloader:
        if len(batch) == 2:
            features, targets = batch
        else:
            *features_list, targets = batch
            features = features_list

        if isinstance(features, (list, tuple)):
            features = [f.to(device) for f in features]
            logits = model(*features)
        else:
            features = features.to(device)
            logits = model(features)

        targets = targets.to(device)
        preds = logits.argmax(dim=1)

        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)

    return compute_intensity_cls_metrics(y_true, y_pred)


@torch.no_grad()
def evaluate_intensity_reg_model(model, dataloader, device: str = "cuda") -> dict:
    """Run an intensity regression model on a dataloader and compute metrics.

    Model outputs (B, 1) predictions.
    """
    model.eval()
    all_preds = []
    all_targets = []

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

        targets = targets.to(device)

        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    y_pred = np.concatenate(all_preds).flatten()
    y_true = np.concatenate(all_targets).flatten()

    return compute_intensity_reg_metrics(y_true, y_pred)


def persistence_baseline(env_data_dicts: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Compute persistence baseline: predict future_direction24 = history_direction24.

    Only uses samples where both history_direction24 and future_direction24 are valid.

    Returns:
        (y_true, y_pred) arrays
    """
    y_true, y_pred = [], []
    for d in env_data_dicts:
        future = d.get("future_direction24", -1)
        hist = d.get("history_direction24", -1)
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

    return np.array(y_true), np.array(y_pred)
