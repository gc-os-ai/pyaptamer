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
        super().__init__()
        self.batchnorm = nn.BatchNorm2d(num_features=1)

    def forward(self, x_apta: Tensor, x_prot: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x_apta, x_prot : Tensor
            Input tensor for aptamers and proteins, respectively. Shapes are
            (batch_size, seq_len_1, n_features) and (batch_size, seq_len_2,
            n_features), respectively.

        Returns
        ----------
        Tensor
            Interaction map tensor of shape (batch_size, 1, seq_len_1, seq_len_2).
        """
        if x_apta.shape[-1] != x_prot.shape[-1]:
            raise ValueError(
                f"Feature dimensions must match. Got {x_apta.shape[-1]} and {x_prot.shape[-1]}."
            )

        # Compute interaction matrix using Einstein summation
        # b = batch_size, i = seq_len_1, j = seq_len_2, d = n_features
        out = torch.einsum('bid,bjd->bij', x_apta, x_prot) 

        # Add channel dimension for compatibility with nn.BatchNorm2d
        out = out.unsqueeze(1)  # (batch_size, 1, s1, s2)

        return self.batchnorm(out)
