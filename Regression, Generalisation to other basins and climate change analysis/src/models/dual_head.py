"""
Stretch goal: Joint direction + intensity prediction with dual heads.

Shares the fusion encoder but has separate classification heads.
"""
import torch
import torch.nn as nn

from src.config import (
    HIDDEN_DIM, NUM_DIRECTION_CLASSES, NUM_INTENSITY_CLASSES, MLP_DROPOUT,
)
from src.models.lstm_1d import LSTMTracker
from src.models.env_temporal import EnvTemporalModel
from src.models.cnn_3d import CNNEncoder3D


class DualHeadFusionModel(nn.Module):
    """Joint direction + intensity prediction via shared encoder.

    Input: (data1d_seq, env_seq, data3d_grid)
    Output: (direction_logits, intensity_logits)
    """

    def __init__(
        self,
        feature_dim: int = HIDDEN_DIM,
        num_dir_classes: int = NUM_DIRECTION_CLASSES,
        num_int_classes: int = NUM_INTENSITY_CLASSES,
        dropout: float = MLP_DROPOUT,
    ):
        super().__init__()

        self.lstm_encoder = LSTMTracker()
        self.env_encoder = EnvTemporalModel()
        self.cnn_encoder = CNNEncoder3D()

        fused_dim = feature_dim * 3
        self.shared = nn.Sequential(
            nn.Linear(fused_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.direction_head = nn.Linear(feature_dim, num_dir_classes)
        self.intensity_head = nn.Linear(feature_dim, num_int_classes)

    def forward(self, data1d, env_data, data3d):
        h_1d = self.lstm_encoder.encode(data1d)
        h_env = self.env_encoder.encode(env_data)
        h_3d = self.cnn_encoder.encode(data3d)

        fused = torch.cat([h_1d, h_env, h_3d], dim=1)
        shared_feat = self.shared(fused)

        dir_logits = self.direction_head(shared_feat)
        int_logits = self.intensity_head(shared_feat)
        return dir_logits, int_logits
