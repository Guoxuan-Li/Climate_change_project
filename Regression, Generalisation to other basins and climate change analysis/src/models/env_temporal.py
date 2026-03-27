"""
Stage 3: Temporal model on Env-Data sequences.

Supports both Transformer and LSTM backends.
"""
import math
import torch
import torch.nn as nn

from src.config import (
    ENV_FEATURE_DIM, HIDDEN_DIM, NUM_DIRECTION_CLASSES,
    TRANSFORMER_HEADS, TRANSFORMER_LAYERS, LSTM_LAYERS, LSTM_DROPOUT, MLP_DROPOUT,
)


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])  # handle odd d_model
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class EnvTemporalModel(nn.Module):
    """Transformer encoder on Env-Data sequences.

    Input: (B, seq_len, 92)
    Output: (B, num_classes) logits
    """

    def __init__(
        self,
        input_dim: int = ENV_FEATURE_DIM,
        d_model: int = HIDDEN_DIM,
        nhead: int = TRANSFORMER_HEADS,
        num_layers: int = TRANSFORMER_LAYERS,
        num_classes: int = NUM_DIRECTION_CLASSES,
        dropout: float = MLP_DROPOUT,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, seq_len, input_dim)"""
        x = self.input_proj(x)     # (B, S, d_model)
        x = self.pos_enc(x)
        x = self.transformer(x)    # (B, S, d_model)
        x = x.mean(dim=1)          # mean pool over time
        return self.classifier(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return feature vector for fusion. (B, d_model)"""
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        return x.mean(dim=1)


class EnvLSTMModel(nn.Module):
    """LSTM alternative for Env-Data sequences.

    Input: (B, seq_len, 92)
    Output: (B, num_classes) logits
    """

    def __init__(
        self,
        input_dim: int = ENV_FEATURE_DIM,
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = LSTM_LAYERS,
        num_classes: int = NUM_DIRECTION_CLASSES,
        dropout: float = LSTM_DROPOUT,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim,
            num_layers=num_layers, dropout=dropout if num_layers > 1 else 0,
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
