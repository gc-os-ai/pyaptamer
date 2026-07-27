"""VAE and CNN_PHMM_VAE model composed from RaptGen layers"""

__author__ = ["nourmajdoub"]
__all__ = ["VAE", "CNN_PHMM_VAE", "CNN_PHMM_VAE_FAST"]


import torch
from torch import nn

from pyaptamer.raptgen.layers._decoder import DecoderPHMM, DecoderPHMM_fast
from pyaptamer.raptgen.layers._encoder import EncoderCNN
from pyaptamer.raptgen.layers._loss import profile_hmm_loss_fn, profile_hmm_loss_fn_fast


class VAE(nn.Module):
    """Variational Autoencoder generic base class for Raptgen VAE variants.

    Implements the base VAE architecture shared by the RaptGen VAE variants. Handles the
    projection from the encoder's hidden representation to the latent space and uses a reparameterization mechanism to sample
    points from the latent distribution for the decoder to generate sequences from.
  
    Parameters
    ----------
    encoder : torch.nn.Module
        The encoder module that maps input data to hidden representations of shape (batch_size,hidden_size), concrete implementation
        in the inheriting class.
        
    decoder : torch.nn.Module
        The decoder module that generates new sequences from a latent point of shape (batch_size,embed_size).

    embed_size : int, optional, default=10
        The dimensionality of the latent space.
        
    hidden_size : int, optional, default=32
        The size of the intermediate hidden representation shared between the encoder output and decoder input layer.


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
    """Raptgen algorithm for unsupervised aptamer sequences generation.
    Implements the raptGen main architecture via a CNN based encoder and profile HMM based decoder.
    
    Parameters
    ----------
   motif_len: int,optional, default=12
       The length of the aptamer sequence being modeled/generated
   
   embed_size : int optional, default=10
       The dimensionality of the latent space.
       
   hidden_size : int, optional, default=32
        The size of the intermediate hidden representation shared between the encoder output and decoder input layer.
        
   kernel_size: int optional, default=7
       Convolution  kernel (window) size used by the CNN encoder, must be an odd number.

    Attributes
    ----------
    encoder : EncoderCNN
        CNN-based encoder that maps the input sequence to hidden space.
    
    decoder : DecoderPHMM
        Profile HMM-based decoder that reconstructs aptamers from the
        learned latent distribution.
       """
    def __init__(self, motif_len=12, embed_size=10, hidden_size=32, kernel_size=7):
        encoder = EncoderCNN(hidden_size, kernel_size)
        decoder = DecoderPHMM(motif_len, embed_size)

        super().__init__(encoder, decoder, embed_size, hidden_size)
        self.loss_fn = profile_hmm_loss_fn


class CNN_PHMM_VAE_FAST(VAE):  # noqa: N801
    """RaptGen algorithm for unsupervised aptamer sequence generation (fast variant).

    Same as `CNN_PHMM_VAE`, but uses `DecoderPHMM_fast` and its matching loss function `profile_hmm_loss_fn_fast` for faster training. 

    Attributes
    ----------
    decoder : DecoderPHMM_fast
        Faster profile HMM-based decoder that reconstructs aptamers from
        the learned latent distribution.
    """
    def __init__(self, motif_len=12, embed_size=10, hidden_size=32, kernel_size=7):
        encoder = EncoderCNN(hidden_size, kernel_size)
        decoder = DecoderPHMM_fast(motif_len, embed_size, hidden_size=hidden_size)

        super().__init__(encoder, decoder, embed_size, hidden_size)
        self.loss_fn = profile_hmm_loss_fn_fast
