"""
Intensity prediction models: classification (4-class) and regression (scalar).

Classification models output (B, 4) logits for intensity change classes:
  0=Strengthening, 1=Str-then-Weak, 2=Weakening, 3=Maintaining

Regression models output (B, 1) scalar delta_wnd_norm (24h wind change
in normalized units). Physical: delta_wind_ms = delta_wnd_norm * 25.

All architectures mirror the existing track prediction models but with
adjusted output dimensions.
"""
import math
import torch
import torch.nn as nn
import torchvision.models as models

from src.config import (
    ENV_FEATURE_DIM, HIDDEN_DIM, MLP_DROPOUT,
    DATA1D_NUM_FEATURES, LSTM_LAYERS, LSTM_DROPOUT,
    TRANSFORMER_HEADS, TRANSFORMER_LAYERS,
    DATA3D_CHANNELS, NUM_INTENSITY_CLASSES,
)


# ═══════════════════════════════════════════════════════════════════════════
#  CLASSIFICATION MODELS  (output: 4 classes)
# ═══════════════════════════════════════════════════════════════════════════

class IntensityClsMLP(nn.Module):
    """MLP for intensity classification from single-timestep Env-Data.

    Input: (B, 92)   Output: (B, 4)
    """

    def __init__(self, input_dim: int = ENV_FEATURE_DIM,
                 hidden_dim: int = HIDDEN_DIM * 2,
                 num_classes: int = NUM_INTENSITY_CLASSES,
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


class IntensityClsLSTM(nn.Module):
    """LSTM for intensity classification from Data1D sequences.

    Input: (B, seq_len, 4)   Output: (B, 4)
    """

    def __init__(
        self,
        input_dim: int = DATA1D_NUM_FEATURES,
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = LSTM_LAYERS,
        dropout: float = LSTM_DROPOUT,
        num_classes: int = NUM_INTENSITY_CLASSES,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(MLP_DROPOUT),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        return self.classifier(h_n[-1])

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]


class _PositionalEncoding(nn.Module):
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


class IntensityClsEnvTemporal(nn.Module):
    """Transformer encoder for intensity classification from Env-Data sequences.

    Input: (B, seq_len, 92)   Output: (B, 4)
    """

    def __init__(
        self,
        input_dim: int = ENV_FEATURE_DIM,
        d_model: int = HIDDEN_DIM,
        nhead: int = TRANSFORMER_HEADS,
        num_layers: int = TRANSFORMER_LAYERS,
        num_classes: int = NUM_INTENSITY_CLASSES,
        dropout: float = MLP_DROPOUT,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = _PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                  num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        return x.mean(dim=1)


class IntensityClsCNN(nn.Module):
    """ResNet-18 backbone for intensity classification from Data3D grids.

    Input: (B, 13, 81, 81)   Output: (B, 4)
    """

    def __init__(
        self,
        in_channels: int = DATA3D_CHANNELS,
        num_classes: int = NUM_INTENSITY_CLASSES,
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
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encode(x)
        return self.classifier(h)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        h = self.pool(h).flatten(1)
        return self.projector(h)


# ═══════════════════════════════════════════════════════════════════════════
#  REGRESSION MODELS  (output: scalar delta_wnd_norm)
# ═══════════════════════════════════════════════════════════════════════════

INTENSITY_REG_OUTPUT_DIM = 1


class IntensityRegMLP(nn.Module):
    """MLP for intensity regression from single-timestep Env-Data.

    Input: (B, 92)   Output: (B, 1)
    """

    def __init__(self, input_dim: int = ENV_FEATURE_DIM,
                 hidden_dim: int = HIDDEN_DIM * 2,
                 output_dim: int = INTENSITY_REG_OUTPUT_DIM,
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
        return self.backbone(x)


class IntensityRegLSTM(nn.Module):
    """LSTM for intensity regression from Data1D sequences.

    Input: (B, seq_len, 4)   Output: (B, 1)
    """

    def __init__(
        self,
        input_dim: int = DATA1D_NUM_FEATURES,
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = LSTM_LAYERS,
        dropout: float = LSTM_DROPOUT,
        output_dim: int = INTENSITY_REG_OUTPUT_DIM,
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
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]


class IntensityRegEnvTemporal(nn.Module):
    """Transformer for intensity regression from Env-Data sequences.

    Input: (B, seq_len, 92)   Output: (B, 1)
    """

    def __init__(
        self,
        input_dim: int = ENV_FEATURE_DIM,
        d_model: int = HIDDEN_DIM,
        nhead: int = TRANSFORMER_HEADS,
        num_layers: int = TRANSFORMER_LAYERS,
        output_dim: int = INTENSITY_REG_OUTPUT_DIM,
        dropout: float = MLP_DROPOUT,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = _PositionalEncoding(d_model)

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
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        return x.mean(dim=1)


class IntensityRegCNN(nn.Module):
    """ResNet-18 backbone for intensity regression from Data3D grids.

    Input: (B, 13, 81, 81)   Output: (B, 1)
    """

    def __init__(
        self,
        in_channels: int = DATA3D_CHANNELS,
        output_dim: int = INTENSITY_REG_OUTPUT_DIM,
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
        h = self.features(x)
        h = self.pool(h).flatten(1)
        return self.projector(h)
