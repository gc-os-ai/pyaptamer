__author__ = ["nennomp"]
__all__ = ["EncoderPredictorConfig", "PositionalEncoding", "TokenPredictor"]

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class EncoderPredictorConfig:
    """
    Hyperparameters' configuration for encoder - token predictor of AptaTrans deep
    neural network.
    """

    num_embeddings: int  # for the embedding layer
    target_dim: int  # for the token predictor
    max_len: int  # for positional encoding


class PositionalEncoding(nn.Module):
    """Positional encoding layer for transformer models.

    Adapted from:
    - https://pytorch-tutorials-preview.netlify.app/beginner/transformer_tutorial.html
    - https://stackoverflow.com/questions/77444485/using-positional-encoding-in-pytorch

    Parameters
    ----------
    d_model : int
        Number of expected input features.
    dropout : float, optional, default=0
        Dropout rate.
    max_len : int, optional, default=5000
        Maximum length of the input sequences.

    Attributes
    ----------
    dropout : nn.Dropout
        Dropout layer.
    pe : Tensor
        Positional encoding tensor of shape (1, `max_len`, `d_model`).
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0,
        max_len: int = 5000,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)  # Changed shape to (1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)  # Changed indexing
        pe[0, :, 1::2] = torch.cos(position * div_term)  # Changed indexing
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch_size, seq_len, n_features (`d_model`)).

        Returns
        -------
        Tensor
            Output tensor of shape (batch_size, seq_len, n_features (`d_model`)), with
            positional encodings applied.
        """
        assert x.shape[1] <= self.max_len, (
            f"Input sequence length {x.shape[1]} exceeds maximum length {self.max_len}."
        )

        out = x + self.pe[:, : x.shape[1], :]
        if self.dropout:
            out = self.dropout(out)
        return out


class TokenPredictor(nn.Module):
    """Token predictor for masked token (mt) and secondary structure (ss) prediction.

    Parameters
    ----------
    d_model : int
        Number of expected input features.
    out_mt, out_ss : int
        Output dimension for masked token and secondary structure prediction,
        respectively.

    Attributes
    ----------
    fc_mt, fc_ss : nn.Linear
        Linear layer for masked token and secondary structure prediction, respectively.
    """

    def __init__(self, d_model: int, d_out_mt: int, d_out_ss: int) -> None:
        """
        Parameters
        ----------
        d_model : int
            Number of expected input features.
        out_mt, out_ss : int
            Output dimension for masked token and secondary structure prediction,
            respectively.
        """
        super().__init__()
        self.fc_mt = nn.Linear(d_model, d_out_mt)
        self.fc_ss = nn.Linear(d_model, d_out_ss)

    def forward(self, x_mt: Tensor, x_ss: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass.

        Parameters
        ----------
        x_mt, x_ss : Tensor
            Input tensor of shape (batch_size, seq_len, n_features `d_model`), for
            masked token and secondary structure prediction, respectively.

        Returns
        ----------
        tuple[Tensor, Tensor]
            A tuple of two output tensors containing the predictions on masked token
            and secondary structure. Shapes are (batch_size, seq_len, n_features
            `d_out_mt`) and (batch_size, seq_len, n_features `d_out_ss`), respectively.
        """
        out_mt = self.fc_mt(x_mt)
        out_ss = self.fc_ss(x_ss)
        return (out_mt, out_ss)
