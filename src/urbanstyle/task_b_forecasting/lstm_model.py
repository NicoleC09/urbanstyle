"""LSTM global con embedding para distinguir categoría y género."""

from __future__ import annotations

import torch
from torch import nn


class GlobalDemandLSTM(nn.Module):
    """Pronostica el siguiente valor normalizado de cualquiera de seis series."""

    def __init__(
        self,
        num_series: int,
        embedding_size: int = 4,
        hidden_size: int = 16,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_series, embedding_size)
        self.lstm = nn.LSTM(
            input_size=1 + embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, windows: torch.Tensor, series_ids: torch.Tensor) -> torch.Tensor:
        """Devuelve un escalar normalizado por ventana de entrada."""
        embeddings = self.embedding(series_ids).unsqueeze(1).expand(-1, windows.shape[1], -1)
        features = torch.cat((windows.unsqueeze(-1), embeddings), dim=-1)
        outputs, _ = self.lstm(features)
        return self.output(outputs[:, -1, :]).squeeze(-1)
