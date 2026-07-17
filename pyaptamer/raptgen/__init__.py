"""RaptGen variational autoencoder with profile HMM decoder for aptamer generation"""


__author__ = ["NoorMajdoub"]
__all__ = ["VAE", "CNN_PHMM_VAE", "CNN_PHMM_VAE_FAST"]

from pyaptamer.raptgen._model import VAE, CNN_PHMM_VAE, CNN_PHMM_VAE_FAST