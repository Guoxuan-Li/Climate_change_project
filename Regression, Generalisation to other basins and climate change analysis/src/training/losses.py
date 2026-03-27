"""
Loss functions for cyclone trajectory prediction.

Includes classification losses (FocalLoss) and regression losses (HaversineLoss).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for multi-class classification with class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: focusing parameter (default 2.0). Higher = more focus on hard examples.
        weight: optional per-class weights tensor of shape (C,).
        reduction: 'mean', 'sum', or 'none'.
    """

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None,
                 reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C) raw scores
            targets: (B,) integer class labels
        """
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        # Gather the log-prob and prob for the true class
        targets_oh = targets.unsqueeze(1)  # (B, 1)
        log_pt = log_probs.gather(1, targets_oh).squeeze(1)  # (B,)
        pt = probs.gather(1, targets_oh).squeeze(1)           # (B,)

        # Focal modulation
        focal_weight = (1 - pt) ** self.gamma

        # Per-class weighting
        if self.weight is not None:
            alpha_t = self.weight[targets]
            focal_weight = focal_weight * alpha_t

        loss = -focal_weight * log_pt

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class HaversineLoss(nn.Module):
    """Haversine (great-circle distance) loss for displacement regression.

    Converts normalized deltas to degree displacements, then computes the
    mean great-circle distance between predicted and actual future positions.

    Normalization: delta_deg = delta_norm * 5 (since norm = raw/50, raw in 0.1deg,
    so delta_deg = delta_norm * 50 / 10 = delta_norm * 5).

    The loss uses a reference latitude of 20N (typical for WP basin) for the
    longitude scaling. For a more precise computation, one would need the actual
    latitude, but this is a reasonable approximation for the loss function.

    Args:
        ref_lat_deg: Reference latitude for cos(lat) correction (default 20.0).
        reduction: 'mean', 'sum', or 'none'.
    """

    EARTH_RADIUS_KM = 6371.0
    NORM_TO_DEG = 5.0  # delta_norm * 5 = delta in degrees

    def __init__(self, ref_lat_deg: float = 20.0, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
        self.ref_lat_rad = math.radians(ref_lat_deg)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, 2) predicted (delta_lon_norm, delta_lat_norm)
            target: (B, 2) actual (delta_lon_norm, delta_lat_norm)
        Returns:
            Haversine distance in km (scalar or per-sample).
        """
        # Convert to degrees
        pred_deg = pred * self.NORM_TO_DEG
        target_deg = target * self.NORM_TO_DEG

        # Convert to radians
        pred_rad = pred_deg * (math.pi / 180.0)
        target_rad = target_deg * (math.pi / 180.0)

        dlon = pred_rad[:, 0] - target_rad[:, 0]
        dlat = pred_rad[:, 1] - target_rad[:, 1]

        # Haversine formula (using reference latitude)
        a = (torch.sin(dlat / 2) ** 2 +
             math.cos(self.ref_lat_rad) * math.cos(self.ref_lat_rad) *
             torch.sin(dlon / 2) ** 2)
        # Clamp for numerical stability
        a = torch.clamp(a, min=0.0, max=1.0)
        c = 2 * torch.asin(torch.sqrt(a))
        dist_km = self.EARTH_RADIUS_KM * c

        if self.reduction == "mean":
            return dist_km.mean()
        elif self.reduction == "sum":
            return dist_km.sum()
        return dist_km
