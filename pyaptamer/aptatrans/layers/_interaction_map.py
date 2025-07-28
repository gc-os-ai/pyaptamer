__author__ = ["nennomp"]
__all__ = ["InteractionMap"]

import torch
import torch.nn as nn
from torch import Tensor


class InteractionMap(nn.Module):
    """Computes the interaction map between aptamers and proteins.

    This module creates a compatibility matrix by computing dot products between
    all pairs of positions from two input sequences, one for aptamers and one for
    proteins. The result is a 2D (interaction) map where each position (i, j)
    corresponds to the interaction strength between the i-th position of the aptamer
    and the j-th position of the protein.

    Attributes
    ----------
    batchnorm : nn.BatchNorm2d
        Batch normalization layer.
    """

    def __init__(self) -> None:
        """
        Parameters
        ----------
        n_features : int
            Number of expected input features.
        """
        super().__init__()
        self.batchnorm = nn.BatchNorm2d(num_features=1)

    def forward(self, x_apta: Tensor, x_prot: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x_apta, x_prot : Tensor
            Input tensor for aptamers and proteins, respectively. Shapes are
            (batch_size, seq_len (s1), n_features) and (batch_size, seq_len (s2),
            n_features), respectively.

        Returns
        ----------
        Tensor
            Interaction map tensor of shape (batch_size, 1, seq_len (s1), seq_len (s2)).
        """
        assert x_apta.shape[-1] == x_prot.shape[-1], (
            "The number of features of `x_apta` and `x_prot` must match."
        )

        # compute interaction matrix using batch matrix multiplication
        out = torch.matmul(x_apta, x_prot.transpose(-2, -1))  # (batch_size, s1, s2)
        # add channel dimension for compatibility with nn.Conv2d operations
        out = out.unsqueeze(1)  # (batch_size, s1, s2) -> (batch_size, 1, s1, s2)

        return self.batchnorm(out)
