"""VAE and CNN_PHMM_VAE model composed from RaptGen layers"""

__author__ = ["nourmajdoub"]
__all__ = ["VAE", "CNN_PHMM_VAE", "CNN_PHMM_VAE_FAST", "VAE"]


import torch
from torch import nn

from pyaptamer.raptgen.layers._decoder import DecoderPHMM, DecoderPHMM_fast
from pyaptamer.raptgen.layers._encoder import EncoderCNN
from pyaptamer.raptgen.layers._loss import profile_hmm_loss_fn, profile_hmm_loss_fn_fast


class VAE(nn.Module):
    def __init__(self, encoder, decoder, embed_size=10, hidden_size=32):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.h2mu = nn.Linear(hidden_size, embed_size)
        self.h2logvar = nn.Linear(hidden_size, embed_size)

    def reparameterize(self, mu, logvar, deterministic=False):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + (std * eps if not deterministic else 0)
        return z

    def forward(self, input, deterministic=False):
        h = self.encoder(input)
        mu = self.h2mu(h)
        logvar = self.h2logvar(h)

        z = self.reparameterize(mu, logvar, deterministic)
        recon_param = self.decoder(z)
        return recon_param, mu, logvar


class CNN_PHMM_VAE(VAE):
    def __init__(self, motif_len=12, embed_size=10, hidden_size=32, kernel_size=7):
        encoder = EncoderCNN(hidden_size, kernel_size)
        decoder = DecoderPHMM(motif_len, embed_size)

        super().__init__(encoder, decoder, embed_size, hidden_size)
        self.loss_fn = profile_hmm_loss_fn


class CNN_PHMM_VAE_FAST(VAE):
    def __init__(self, motif_len=12, embed_size=10, hidden_size=32, kernel_size=7):
        encoder = EncoderCNN(hidden_size, kernel_size)
        decoder = DecoderPHMM_fast(motif_len, embed_size, hidden_size=hidden_size)

        super().__init__(
            encoder, decoder, embed_size, hidden_size
        )
        self.loss_fn = profile_hmm_loss_fn_fast
