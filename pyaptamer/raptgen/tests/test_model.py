"""Tests for the raptgen CNN_PHMM_VAE model"""

__author__ = ["NoorMajdoub"]


import pytest
import torch
from pyaptamer.raptgen._model import CNN_PHMM_VAE
from pyaptamer.raptgen.layers._encoder import EncoderCNN
from pyaptamer.raptgen.layers._decoder import DecoderPHMM
from pyaptamer.raptgen.layers._loss import profile_hmm_loss_fn

@pytest.mark.parametrize(
    "motif_len, embed_size, hidden_size, kernel_size", [(4, 8, 16, 5), (10, 16, 32, 7)]
)
def test_cnn_phmm_vae_layers(motif_len, embed_size, hidden_size, kernel_size):
    """
    Checks that `CNN_PHMM_VAE` builds the correct encoder/decoder and loss function.
    """
    model = CNN_PHMM_VAE(
        motif_len=motif_len,
        embed_size=embed_size,
        hidden_size=hidden_size,
        kernel_size=kernel_size,
    )

    assert isinstance(model.encoder, EncoderCNN)
    assert isinstance(model.decoder, DecoderPHMM)
    assert model.loss_fn is profile_hmm_loss_fn

    assert model.h2mu.out_features == embed_size
    assert model.h2logvar.out_features == embed_size


@pytest.mark.parametrize(
    "motif_len, embed_size, hidden_size, kernel_size, batch_size, seq_len",
    [(4, 8, 16, 5, 3, 20), (10, 16, 32, 7, 2, 40)],
)
def test_cnn_phmm_vae_forward(
    motif_len, embed_size, hidden_size, kernel_size, batch_size, seq_len
):
    """
    Tests the forward pass of CNN_PHMM_VAE.
    """
    model = CNN_PHMM_VAE(
        motif_len=motif_len,
        embed_size=embed_size,
        hidden_size=hidden_size,
        kernel_size=kernel_size,
    )

    x = torch.randint(low=0, high=4, size=(batch_size, seq_len))

    recon_param, mu, logvar = model(x)
    transition_proba, emission_proba = recon_param

    assert mu.shape == (batch_size, embed_size)
    assert logvar.shape == (batch_size, embed_size)
    assert transition_proba.shape == (batch_size, motif_len + 1, 7)
    assert emission_proba.shape == (batch_size, motif_len, 4)
