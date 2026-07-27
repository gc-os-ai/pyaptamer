"""Decoder for RaptGen's variational autoencoder"""

__author__ = ["NoorMajdoub"]
__all__ = ["DecoderPHMM", "DecoderPHMM_fast"]

import torch
from torch import nn


class DecoderPHMM(nn.Module):
    """ 
    RaptGen profile HMM decoder for aptamer unsupervised generation.
    Reconstructs/generates aptamer sequences from a latent point sampled
from the encoder's learned distribution. Works by predicting the
parameters of a profile Hidden Markov Model (HMM), per-position transition and
emission probabilities of the nucleotides for the entire sequence in a forward pass.
    Parameters
    ---------- 
    motif_len : int 
        The length of the aptamer sequences being modeled/generated. 
    
    embed_size : int 
        The dimensionality of the input latent space. 
    
    hidden_size : int, optional, default=32 
        The size of the shared hidden representation.

    Attributes 
    ---------- 

    fc1 : nn.Sequential 
        Projects the latent point to the shared hidden representation. 
    
    tr_from_M, tr_from_I, tr_from_D : nn.Sequential 
        Predict transition probabilities out of the Match, Insert, and Delete states depending on how many transitions are possible from that state. 
    
    emission : nn.Sequential 
        Predicts per-position emission probabilities over the 4 nucleotides.
    """
    def __init__(self, motif_len, embed_size, hidden_size=32):
        super().__init__()

        class View(nn.Module):
            def __init__(self, shape):
                super().__init__()
                self.shape = shape

            def forward(self, x):
                return x.view(*self.shape)

        self.fc1 = nn.Sequential(
            nn.Linear(embed_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

        self.tr_from_M = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(hidden_size, (motif_len + 1) * 3),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            View((-1, motif_len + 1, 3)),
            nn.LogSoftmax(dim=2),
        )
        self.tr_from_I = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(hidden_size, (motif_len + 1) * 2),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            View((-1, motif_len + 1, 2)),
            nn.LogSoftmax(dim=2),
        )
        self.tr_from_D = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(hidden_size, (motif_len + 1) * 2),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            View((-1, motif_len + 1, 2)),
            nn.LogSoftmax(dim=2),
        )

        self.emission = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(hidden_size, motif_len * 4),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            View((-1, motif_len, 4)),
            nn.LogSoftmax(dim=2),
        )

    def forward(self, input):
        x = self.fc1(input)

        transition_from_match = self.tr_from_M(x)
        transition_from_insertion = self.tr_from_I(x)
        transition_from_deletion = self.tr_from_D(x)

        emission_proba = self.emission(x)
        return (
            torch.cat(
                (
                    transition_from_match,
                    transition_from_insertion,
                    transition_from_deletion,
                ),
                dim=2,
            ),
            emission_proba,
        )


class DecoderPHMM_fast(nn.Module):  # noqa: N801
    """
    Optimized version of DecoderPHMM, but computes all state
    transitions (Match/Insert/Delete) as a single combined 3x3 transition 
    tensor per position, instead of three separate transition sub-networks.
    """
    def __init__(self, motif_len, embed_size, hidden_size=32):
        super().__init__()

        class View(nn.Module):
            def __init__(self, shape):
                super().__init__()
                self.shape = shape

            def forward(self, x):
                return x.view(*self.shape)

        self.fc = nn.Sequential(
            nn.Linear(embed_size, hidden_size),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

        self.transition = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(hidden_size, 3 * 3 * (motif_len + 1)),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            View((-1, 3, 3, motif_len + 1)),
            nn.LogSoftmax(dim=2),
        )

        self.emission = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(hidden_size, motif_len * 4),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            View((-1, motif_len, 4)),
            nn.LogSoftmax(dim=2),
        )

    def forward(self, input):
        x = self.fc(input)

        transition_proba = self.transition(x)
        emission_proba = self.emission(x)

        return (transition_proba, emission_proba)
