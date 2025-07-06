__author__ = ["nennomp"]
__all__ = ["EmbeddingConfig", "PositionalEncoding", "TokenPredictor"]

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class EmbeddingConfig:
    n_vocabs: int
    n_target_vocabs: int
    max_len: int


class PositionalEncoding(nn.Module):
    # Adapted from
    # https://stackoverflow.com/questions/77444485/using-positional-encoding-in-pytorch
    """
    Attributes:
        pe: Positional encoding.
    """
    def __init__(self, d_model: int, dropout: Optional[float] = None, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout is not None else None

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        out = x + self.pe[:x.size(0)]
        if self.dropout:
            out = self.dropout(out)
        return out


class TokenPredictor(nn.Module):
    def __init__(self, d_model: int, n_vocabs: int, n_target_vocabs: int) -> None:
        super().__init__()
        self.fc_mt = nn.Linear(d_model, n_vocabs) # masked tokens prediction
        self.fc_ss = nn.Linear(d_model, n_target_vocabs) # secondary structure (ss) prediction

    def forward(self, x_mt: Tensor, x_ss: Tensor) -> tuple[Tensor, Tensor]:
        out_mt = self.fc_mt(x_mt)
        out_ss = self.fc_ss(x_ss)
        return (out_mt, out_ss)