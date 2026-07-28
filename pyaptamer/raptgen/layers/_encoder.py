"""Encoder for RaptGen's variational autoencoder"""

__author__ = ["NoorMajdoub"]
__all__ = ["EncoderCNN"]

from torch import nn
from torch.nn import functional as F

from pyaptamer.raptgen.layers._conv import Bottleneck


class EncoderCNN(nn.Module):
    """
    RaptGen CNN-based encoder for mapping aptamer sequences to a hidden representation.
    Embeds each nucleotide (A, T, G, C) into a learned vector using a
    learned embedding lookup table, passes the sequence through a stack of
    residual `Bottleneck` blocks to extract sequence motifs , then max-pools over
    the sequence length to produce a single fixed-size hidden representation
    per sequence.

    Parameters
    ----------

    embedding_dim : int, optional, default=32
        The dimensionality of the nucleotide embedding used throughout
        the residual blocks.
        Also the size of the output hidden representation.

    window_size : int, optional, default=7
        Convolution kernel size passed to each `Bottleneck` block. Must be odd.

    num_layers : int, optional, default=6
        Number of stacked `Bottleneck` residual blocks.

    Attributes
    ----------
        embed : nn.Embedding
            Learned nucleotide embedding of 4 tokens (A, T, G, C).

        resnet : nn.Sequential
            Stack of `num_layers` `Bottleneck` blocks.
    """

    def __init__(self, embedding_dim=32, window_size=7, num_layers=6):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.window_size = window_size

        self.embed = nn.Embedding(
            num_embeddings=4,  # [A,T,G,C]
            embedding_dim=embedding_dim,
        )

        modules = [Bottleneck(embedding_dim, window_size) for _ in range(num_layers)]
        self.resnet = nn.Sequential(*modules)

    def forward(self, seqences):
        # change X from (N, L) to (N, L, C)
        x = F.leaky_relu(self.embed(seqences))

        # change X to (N, C, L)
        x = x.transpose(1, 2)
        value, indices = self.resnet(x).max(dim=2)
        return value
