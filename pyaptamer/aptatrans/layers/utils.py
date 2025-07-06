__author__ = ["nennomp"]
__all__ = ["InteractionMap", "InteractionMapv2"]

import torch
import torch.nn as nn
from torch import Tensor


class InteractionMap(nn.Module):
    # https://github.com/PNUMLB/AptaTrans/blob/master/encoders.py
    """
    Computes interaction maps between two vectors, one for aptamers and one for proteins. The module creates a normalized interaction/similarity matrix between the two vectors.
    
    TODO: make sure behaviour corresponds to '_v2'.
    """
    def __init__(self) -> None:
        super().__init__()
        self.batchnorm = nn.BatchNorm2d(1)

    def forward(self, x_apta: Tensor, x_prot: Tensor) -> Tensor:
        """
        Args:
            x_apta: Vector of aptamer features, shape (batch_size, seq_len_1, embedding_dim).
            x_prot: Vector of protein features, shape (batch_size, seq_len_2, embedding_dim).
            
        Returns:
            Normalized interaction map of shape (batch_size, 1, seq_len_1, seq_len_2).
        """
        # Compute pairwise dot products using Einstein summation
        # 'b i d, b j d -> b i j' means:
        # - For each batch element b
        # - Take vector at position i from x_apta (dimension d)
        # - Take vector at position j from x_prot (dimension d)
        # - Compute dot product (sum over d)
        # Result: (batch, seq_len_1, seq_len_2)
        out = torch.einsum('b i d, b j d -> b i j', x_apta, x_prot)
        
        # Add channel dimension for compatibility with Conv2d operations
        # (batch, seq_len_1, seq_len_2) -> (batch, 1, seq_len_1, seq_len_2)
        out = torch.unsqueeze(out, 1)
        out = self.batchnorm(out)
        
        return out


class InteractionMapv2(nn.Module):
    # TODO: make sure behaviour corresponds to original.
    def __init__(self) -> None:
        super().__init__()
        # BatchNorm2d for normalizing the single-channel interaction map
        self.batchnorm = nn.BatchNorm2d(1)

    def forward(self, x_apta: Tensor, x_prot: Tensor) -> Tensor:
        """
        Args:
            x_apta: Vector of aptamer features, shape (batch_size, seq_len_1, embedding_dim).
            x_prot: Vector of protein features, shape (batch_size, seq_len_2, embedding_dim).
            
        Returns:
            Normalized interaction map of shape (batch_size, 1, seq_len_1, seq_len_2).
        """
        # Compute interaction matrix using batch matrix multiplication
        # Equivalent to: x_apta @ x_prot.transpose(-2, -1)
        out = torch.matmul(x_apta, x_prot.transpose(-2, -1)) # shape (batch, seq_len_1, seq_len_2)
        
        # Add channel dimension for compatibility with Conv2d operations
        # (batch, seq_len_1, seq_len_2) -> (batch, 1, seq_len_1, seq_len_2)
        out = out.unsqueeze(1)
        out = self.batchnorm(out)
        
        return out