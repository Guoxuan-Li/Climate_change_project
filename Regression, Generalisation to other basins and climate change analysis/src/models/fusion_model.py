"""
Stage 5: Multimodal late-fusion model.

Combines encoders from Stages 2, 3, and 4 with a fusion MLP head.
"""
import torch
import torch.nn as nn

from src.config import (
    DATA1D_NUM_FEATURES, ENV_FEATURE_DIM, DATA3D_CHANNELS,
    HIDDEN_DIM, NUM_DIRECTION_CLASSES, MLP_DROPOUT,
    LSTM_LAYERS, LSTM_DROPOUT, TRANSFORMER_HEADS, TRANSFORMER_LAYERS,
)
from src.models.lstm_1d import LSTMTracker
from src.models.env_temporal import EnvTemporalModel
from src.models.cnn_3d import CNNEncoder3D


class FusionModel(nn.Module):
    """Late fusion of Data1D (LSTM) + Env-Data (Transformer) + Data3D (CNN).

    Input: (data1d_seq, env_seq, data3d_grid)
    Output: (B, num_classes) logits
    """

    def __init__(
        self,
        feature_dim: int = HIDDEN_DIM,
        num_classes: int = NUM_DIRECTION_CLASSES,
        dropout: float = MLP_DROPOUT,
    ):
        super().__init__()

        # Branch encoders (without their classification heads)
        self.lstm_encoder = LSTMTracker()
        self.env_encoder = EnvTemporalModel()
        self.cnn_encoder = CNNEncoder3D()

        # Fusion head: 3 * feature_dim -> num_classes
        fused_dim = feature_dim * 3
        self.fusion_head = nn.Sequential(
            nn.Linear(fused_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, data1d: torch.Tensor, env_data: torch.Tensor,
                data3d: torch.Tensor) -> torch.Tensor:
        """
        Args:
            data1d: (B, seq_len, 4)
            env_data: (B, seq_len, 92)
            data3d: (B, 13, 81, 81)
        """
        h_1d = self.lstm_encoder.encode(data1d)    # (B, hidden)
        h_env = self.env_encoder.encode(env_data)   # (B, hidden)
        h_3d = self.cnn_encoder.encode(data3d)      # (B, hidden)

        fused = torch.cat([h_1d, h_env, h_3d], dim=1)  # (B, 3*hidden)
        return self.fusion_head(fused)


class FusionModel2Branch(nn.Module):
    """Two-branch fusion: Data1D (LSTM) + Env-Data (Transformer).

    For ablation without Data3D.
    """

    def __init__(
        self,
        feature_dim: int = HIDDEN_DIM,
        num_classes: int = NUM_DIRECTION_CLASSES,
        dropout: float = MLP_DROPOUT,
    ):
        super().__init__()
        self.lstm_encoder = LSTMTracker()
        self.env_encoder = EnvTemporalModel()

        fused_dim = feature_dim * 2
        self.fusion_head = nn.Sequential(
            nn.Linear(fused_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, data1d: torch.Tensor, env_data: torch.Tensor) -> torch.Tensor:
        h_1d = self.lstm_encoder.encode(data1d)
        h_env = self.env_encoder.encode(env_data)
        fused = torch.cat([h_1d, h_env], dim=1)
        return self.fusion_head(fused)
