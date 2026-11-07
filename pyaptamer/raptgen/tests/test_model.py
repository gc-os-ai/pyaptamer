"""Tests for the raptgen CNNPHMMVAE model"""

__author__ = ["NoorMajdoub"]


import pytest
import torch

from pyaptamer.raptgen._model import CNNPHMMVAE, CNNPHMMVAEFast
from pyaptamer.raptgen.layers._decoder import DecoderPHMM
from pyaptamer.raptgen.layers._encoder import EncoderCNN
from pyaptamer.raptgen.layers._loss import profile_hmm_loss_fn


@pytest.mark.parametrize(
    "motif_len, embed_size, hidden_size, kernel_size", [(4, 8, 16, 5), (10, 16, 32, 7)]
)
def test_cnn_phmm_vae_layers(motif_len, embed_size, hidden_size, kernel_size):
    """
    Checks that `CNNPHMMVAE` builds the correct encoder/decoder and loss function.
    """
    model = CNNPHMMVAE(
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
    Tests the forward pass of CNNPHMMVAE.
    """
    model = CNNPHMMVAE(
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


@pytest.mark.parametrize("model_cls", [CNNPHMMVAE, CNNPHMMVAEFast])
def test_cnn_phmm_vae_training_step(model_cls):
    """One forward, loss and backward pass yields a finite loss and gradients."""
    torch.manual_seed(0)
    model = model_cls(motif_len=4, embed_size=8, hidden_size=16, kernel_size=5)
    x = torch.randint(low=0, high=4, size=(3, 12))

    recon_param, mu, logvar = model(x)
    loss = model.loss_fn(x, recon_param, mu, logvar)
    loss.backward()

    assert torch.isfinite(loss)
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
