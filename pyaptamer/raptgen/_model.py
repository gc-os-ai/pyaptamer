"""VAE and CNN_PHMM_VAE model composed from RaptGen layers"""

__author__ = ["nourmajdoub"]
__all__ = ["VAE", "CNN_PHMM_VAE", "CNN_PHMM_VAE_FAST", "VAE"]


import torch
from torch import nn

from pyaptamer.raptgen.layers._decoder import DecoderPHMM, DecoderPHMM_fast
from pyaptamer.raptgen.layers._encoder import EncoderCNN
from pyaptamer.raptgen.layers._loss import profile_hmm_loss_fn, profile_hmm_loss_fn_fast


class VAE(nn.Module):
    """Variational Autoencoder generic base class for Raptgen VAE variants.

    Implements the base VAE architecture shared by the RaptGen VAE variants, handles the
    projection from the encoders hidden representation to the latent space, reconstructs points in the latent space 
    via reparameterization mecanism (mean, variance and added random noise) to be used by the decoder for generation.

    Parameters
    ----------
    encoder : torch.nn.Module
        The encoder module that maps input data to hidden representations of shape (batch_size,hidden_size),concrete implementation
        in the inheritent class
        
    decoder : torch.nn.Module
        The decoder module that reconstructs/generate new sequences from the latent space of shape (batch size,embed size

    embed_size : int, optional, default=10
        The dimensionality of the generated sequence

    hidden_size : int, optional, default=32
        The size of the intermediate hidden reresentaation shared between the encoder output and decoder input layer

    Attributes
    ----------
    encoder : torch.nn.Module
        The encoder module.

    decoder : torch.nn.Module
        The decoder module.

    h2mu : torch.nn.Linear
        Linear layer that maps hidden representation to mean (mu) of the 
        latent distribution. Maps from `hidden_size` to `embed_size`.

    h2logvar : torch.nn.Linear
        Linear layer that maps hidden representation to log-variance (logvar)
        of the latent distribution. Maps from `hidden_size` to `embed_size`.
    """
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


class CNN_PHMM_VAE(VAE):  # noqa: N801
    """Raptgen algorithm for unsupervsed aptamer seqeunces generation.
    Implements the raptegen main architecture via a CNN based encoder and profile HMM based decoder.
    
    Parameters
    ----------
   motif len: int,optional 
   Length of the aptamer squence being modeled/generated
   
   embed size : int optional 
   kernel size: Convolution  kernel (window) size usd in the cnn encoder,must be an odd number

    Attributes
    ----------
       encoder the cnn based encoder that maps the input seqeunce to hidden space
       decode the hmm based decoder that reconstructs aptamer from the learned distribution
       """
    def __init__(self, motif_len=12, embed_size=10, hidden_size=32, kernel_size=7):
        encoder = EncoderCNN(hidden_size, kernel_size)
        decoder = DecoderPHMM(motif_len, embed_size)

        super().__init__(encoder, decoder, embed_size, hidden_size)
        self.loss_fn = profile_hmm_loss_fn


class CNN_PHMM_VAE_FAST(VAE):  # noqa: N801
    """RaptGen algorithm for unsupervised aptamer sequence generation.
    Same as `CNN_PHMM_VAE`, but uses `DecoderPHMM_fast` and its matching
    loss function `profile_hmm_loss_fn_fast` for faster training.
    """
    def __init__(self, motif_len=12, embed_size=10, hidden_size=32, kernel_size=7):
        encoder = EncoderCNN(hidden_size, kernel_size)
        decoder = DecoderPHMM_fast(motif_len, embed_size, hidden_size=hidden_size)

        super().__init__(encoder, decoder, embed_size, hidden_size)
        self.loss_fn = profile_hmm_loss_fn_fast
