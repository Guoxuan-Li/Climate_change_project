"""
Stage 2: LSTM model on Data1D (track coordinate) sequences.
"""
import torch
import torch.nn as nn

from src.config import (
    DATA1D_NUM_FEATURES, HIDDEN_DIM, LSTM_LAYERS, LSTM_DROPOUT,
    NUM_DIRECTION_CLASSES, MLP_DROPOUT,
)


class LSTMTracker(nn.Module):
    """LSTM encoder for Data1D track sequences.

    Input: (B, seq_len, 4) — normalized [LONG, LAT, PRES, WND]
    Output: (B, num_classes) logits
    """

    def __init__(
        self,
        input_dim: int = DATA1D_NUM_FEATURES,
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = LSTM_LAYERS,
        dropout: float = LSTM_DROPOUT,
        num_classes: int = NUM_DIRECTION_CLASSES,
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
        """x: (B, seq_len, input_dim)"""
        _, (h_n, _) = self.lstm(x)  # h_n: (num_layers, B, hidden)
        last_hidden = h_n[-1]        # (B, hidden)
        return self.classifier(last_hidden)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the feature vector (for fusion). (B, hidden_dim)"""
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]
