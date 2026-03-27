"""
Stage 1: MLP baseline on single-timestep Env-Data features.
"""
import torch
import torch.nn as nn

from src.config import ENV_FEATURE_DIM, NUM_DIRECTION_CLASSES, HIDDEN_DIM, MLP_DROPOUT


class BaselineMLP(nn.Module):
    """3-layer MLP for direction classification from Env-Data features.

    Input: (B, 92) feature vector
    Output: (B, 8) logits
    """

    def __init__(self, input_dim: int = ENV_FEATURE_DIM,
                 hidden_dim: int = HIDDEN_DIM * 2,
                 num_classes: int = NUM_DIRECTION_CLASSES,
                 dropout: float = MLP_DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
