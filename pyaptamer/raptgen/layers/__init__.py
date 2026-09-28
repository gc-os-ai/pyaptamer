"""Architectural components for RaptGen's variational autoencoder"""

__author__ = ["NoorMajdoub"]
__all__ = [
    "InvertedBottleneck",
    "EncoderCNN",
    "DecoderPHMM",
    "DecoderPHMMFast",
    "ProfileHMMSampler",
    "NucleotideIndex",
    "State",
    "Transition",
    "profile_hmm_loss",
    "profile_hmm_loss_fn",
    "profile_hmm_loss_fn_fast",
    "torch_multi_polytope_dp_log",
    "kld_loss",
]

from pyaptamer.raptgen.layers._conv import InvertedBottleneck
from pyaptamer.raptgen.layers._decoder import DecoderPHMM, DecoderPHMMFast
from pyaptamer.raptgen.layers._encoder import EncoderCNN
from pyaptamer.raptgen.layers._loss import (
    kld_loss,
    profile_hmm_loss,
    profile_hmm_loss_fn,
    profile_hmm_loss_fn_fast,
    torch_multi_polytope_dp_log,
)
from pyaptamer.raptgen.layers._sampler import ProfileHMMSampler
from pyaptamer.raptgen.layers._utils import NucleotideIndex, State, Transition
