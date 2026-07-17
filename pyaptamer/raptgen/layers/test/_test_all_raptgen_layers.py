"""   Tests for the raptgen.layers module """
__author__ = ["NoorMajdoub"]
__all__ = ["test_bottleneck_layers", "test_bottleneck_forward", "test_encodercnn_layers", "test_encodercnn_forward","test_decoderphmm_layers","test_decoderphmm_forward","test_cnn_phmm_vae_layers","test_cnn_phmm_vae_forward"]

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from pyaptamer.raptgen._model import CNN_PHMM_VAE
from pyaptamer.raptgen.layers._conv import Bottleneck
from pyaptamer.raptgen.layers._loss import profile_hmm_loss_fn
from pyaptamer.raptgen.layers._encoder import EncoderCNN

@pytest.mark.parametrize(
        "init_dim, window_size", [(8, 3), (16, 5), (32, 7)]
        )

def test_bottleneck_layers(init_dim, window_size):
    """Check Bottleneck() initializes its conv and batchnorm layers correctly"""
    block = Bottleneck(init_dim=init_dim, window_size=window_size)

    assert isinstance(block.conv1, nn.Conv1d)
    assert block.conv1.in_channels == init_dim
    assert block.conv1.out_channels == init_dim * 2
    assert block.conv1.kernel_size == (1,)

    assert isinstance(block.conv2, nn.Conv1d)
    assert block.conv2.in_channels == init_dim * 2
    assert block.conv2.out_channels == init_dim * 2
    assert block.conv2.kernel_size == (window_size,)

  
    assert isinstance(block.conv3, nn.Conv1d)
    assert block.conv3.in_channels == init_dim * 2
    assert block.conv3.out_channels == init_dim
    assert block.conv3.kernel_size == (1,)

    assert isinstance(block.bn1, nn.BatchNorm1d)
    assert isinstance(block.bn2, nn.BatchNorm1d)
    assert isinstance(block.bn3, nn.BatchNorm1d)



@pytest.mark.parametrize(
    "init_dim, window_size, x",    
    [
        (8, 3, torch.randn(2, 8, 50)),      
        (16, 5, torch.randn(4, 16, 100)),   
        (32, 7, torch.randn(1, 32, 200)),   
    ],
)
def test_bottleneck_forward(init_dim, window_size,x):
    """
    Test the forward pass of the Bottleneck residual block (Shape must not change)
    """
    block = Bottleneck(init_dim=init_dim, window_size=window_size)
    out = block(x)
    assert out.shape == x.shape

def test_bottleneck_rejects_even_window_size():
    """
    Check that Bottleneck enforces odd window_size
    """
    with pytest.raises(AssertionError):
        Bottleneck(init_dim=8, window_size=4)





@pytest.mark.parametrize(
    "embedding_dim, window_size, num_layers", [(16, 3, 2), (32, 7, 6)]
)
def test_encodercnn_layers(embedding_dim, window_size, num_layers):
    """
    Check EncoderCNN embedding and layers initialization
    """
    encoder = EncoderCNN(
        embedding_dim=embedding_dim, window_size=window_size, num_layers=num_layers
    )

    assert isinstance(encoder.embed, nn.Embedding)
    assert encoder.embed.num_embeddings == 4
    assert encoder.embed.embedding_dim == embedding_dim

    assert isinstance(encoder.resnet, nn.Sequential)
    assert len(encoder.resnet) == num_layers
    assert all(isinstance(layer, Bottleneck) for layer in encoder.resnet)


@pytest.mark.parametrize(
    "embedding_dim, num_layers, batch_size, seq_len",
    [(16, 2, 4, 30), (32, 6, 2, 50)],
)
def test_encodercnn_forward(embedding_dim, num_layers, batch_size, seq_len):
    """
    Test the forward pass of EncoderCNN (Output shape)
    """
    encoder = EncoderCNN(embedding_dim=embedding_dim, num_layers=num_layers)
    x = torch.randint(low=0, high=4, size=(batch_size, seq_len))

    out = encoder(x)
    assert out.shape == (batch_size, embedding_dim)





from pyaptamer.raptgen.layers._decoder import DecoderPHMM

@pytest.mark.parametrize(
    "motif_len, embed_size, hidden_size", [(4, 8, 16), (10, 16, 32)]
)
def test_decoderphmm_layers(motif_len, embed_size,  hidden_size):
    """
    Check DecoderPHMM initializes its layers correctly
    """
    decoder = DecoderPHMM(
        motif_len=motif_len, embed_size=embed_size, hidden_size=hidden_size
    )

    assert isinstance(decoder.fc1, nn.sequential)
    assert decoder.fc1[0].in_features == embed_size
    assert decoder.fc1[0].out_features == hidden_size



@pytest.mark.parametrize(
    "motif_len, embed_size, hidden_size, batch_size",
    [(4, 8, 16, 2), (10, 16, 32, 5)],
)
def test_decoderphmm_forward(motif_len, embed_size, hidden_size, batch_size):
    """
    Test the forward pass of DecoderPHMM (check output shapes)
    """
    decoder = DecoderPHMM(motif_len=motif_len, embed_size=embed_size, hidden_size=hidden_size)
    x = torch.randn(batch_size, embed_size)

    transition_proba, emission_proba = decoder(x)

    assert transition_proba.shape == (batch_size, motif_len + 1, 7)
    assert emission_proba.shape == (batch_size, motif_len, 4)



@pytest.mark.parametrize(
    "motif_len, embed_size, hidden_size, kernel_size", [(4, 8, 16, 5), (10, 16, 32, 7)]
)
def test_cnn_phmm_vae_layers(motif_len, embed_size, hidden_size, kernel_size):
    """
    Check CNN_PHMM_VAE builds the correct encoder/decoder and loss_fn
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
def test_cnn_phmm_vae_forward(motif_len, embed_size, hidden_size, kernel_size, batch_size, seq_len):
    """
    Test the forward pass of CNN_PHMM_VAE
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