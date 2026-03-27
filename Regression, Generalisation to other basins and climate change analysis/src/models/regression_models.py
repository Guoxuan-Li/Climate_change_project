"""
Regression model variants: predict (delta_lon_norm, delta_lat_norm) as 2 continuous values.

Each model mirrors its classification counterpart but outputs (B, 2) instead of (B, 8).
All models include an encode() method returning the feature vector for potential fusion.
"""
import math
import torch
import torch.nn as nn
import torchvision.models as models

from src.config import (
    ENV_FEATURE_DIM, HIDDEN_DIM, MLP_DROPOUT,
    DATA1D_NUM_FEATURES, LSTM_LAYERS, LSTM_DROPOUT,
    TRANSFORMER_HEADS, TRANSFORMER_LAYERS,
    DATA3D_CHANNELS,
)

REG_OUTPUT_DIM = 2  # (delta_lon_norm, delta_lat_norm)


class RegMLP(nn.Module):
    """3-layer MLP for regression from Env-Data features.

    Input: (B, 92) feature vector
    Output: (B, 2) predicted displacement
    """

    def __init__(self, input_dim: int = ENV_FEATURE_DIM,
                 hidden_dim: int = HIDDEN_DIM * 2,
                 output_dim: int = REG_OUTPUT_DIM,
                 dropout: float = MLP_DROPOUT):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the feature vector (B, hidden_dim//2)."""
        return self.backbone(x)


class RegLSTM(nn.Module):
    """LSTM encoder for Data1D track sequences, regression output.

    Input: (B, seq_len, 4)
    Output: (B, 2) predicted displacement
    """

    def __init__(
        self,
        input_dim: int = DATA1D_NUM_FEATURES,
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = LSTM_LAYERS,
        dropout: float = LSTM_DROPOUT,
        output_dim: int = REG_OUTPUT_DIM,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Dropout(MLP_DROPOUT),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1])

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the feature vector (B, hidden_dim)."""
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class RegEnvTemporal(nn.Module):
    """Transformer encoder on Env-Data sequences, regression output.

    Input: (B, seq_len, 92)
    Output: (B, 2) predicted displacement
    """

    def __init__(
        self,
        input_dim: int = ENV_FEATURE_DIM,
        d_model: int = HIDDEN_DIM,
        nhead: int = TRANSFORMER_HEADS,
        num_layers: int = TRANSFORMER_LAYERS,
        output_dim: int = REG_OUTPUT_DIM,
        dropout: float = MLP_DROPOUT,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                  num_layers=num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.head(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return feature vector (B, d_model)."""
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        return x.mean(dim=1)


class RegCNN3D(nn.Module):
    """ResNet-18 backbone for Data3D atmospheric grids, regression output.

    Input: (B, 13, 81, 81)
    Output: (B, 2) predicted displacement
    """

    def __init__(
        self,
        in_channels: int = DATA3D_CHANNELS,
        output_dim: int = REG_OUTPUT_DIM,
        feature_dim: int = HIDDEN_DIM,
    ):
        super().__init__()
        resnet = models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2,
                                 padding=3, bias=False)

        self.features = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

        resnet_out_dim = 512
        self.projector = nn.Sequential(
            nn.Linear(resnet_out_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(MLP_DROPOUT),
        )
        self.head = nn.Linear(feature_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encode(x)
        return self.head(h)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return feature vector (B, feature_dim)."""
        h = self.features(x)
        h = self.pool(h).flatten(1)
        return self.projector(h)


class RegFusionModel(nn.Module):
    """Late fusion of Data1D + Env-Data + Data3D, regression output.

    Input: (data1d_seq, env_seq, data3d_grid)
    Output: (B, 2) predicted displacement
    """

    def __init__(
        self,
        feature_dim: int = HIDDEN_DIM,
        output_dim: int = REG_OUTPUT_DIM,
        dropout: float = MLP_DROPOUT,
    ):
        super().__init__()

        self.lstm_encoder = RegLSTM()
        self.env_encoder = RegEnvTemporal()
        self.cnn_encoder = RegCNN3D()

        fused_dim = feature_dim * 3
        self.fusion_head = nn.Sequential(
            nn.Linear(fused_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, output_dim),
        )

    def forward(self, data1d: torch.Tensor, env_data: torch.Tensor,
                data3d: torch.Tensor) -> torch.Tensor:
        h_1d = self.lstm_encoder.encode(data1d)
        h_env = self.env_encoder.encode(env_data)
        h_3d = self.cnn_encoder.encode(data3d)

        fused = torch.cat([h_1d, h_env, h_3d], dim=1)
        return self.fusion_head(fused)

    def encode(self, data1d: torch.Tensor, env_data: torch.Tensor,
               data3d: torch.Tensor) -> torch.Tensor:
        """Return fused feature vector (B, 3*feature_dim)."""
        h_1d = self.lstm_encoder.encode(data1d)
        h_env = self.env_encoder.encode(env_data)
        h_3d = self.cnn_encoder.encode(data3d)
        return torch.cat([h_1d, h_env, h_3d], dim=1)
