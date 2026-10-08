"""End-to-end AptaDiff algorithm taking SELEX reads in, generating aptamer sequences."""

__author__ = ["aditi-dsi"]
__all__ = ["AptaDiff"]

import numpy as np
import pandas as pd
import torch
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from sklearn.utils import Tags


class AptaDiff(BaseEstimator):
    """Design and optimise aptamers with AptaDiff [1]_.

    Parameters
    ----------
    vae : object
    trimmer : BaseTransform or None, optional, default=None
    n_clusters : int, optional, default=8
    gmm_restarts : int, optional, default=100
    generator : AptaDiffGenerator or None, optional, default=None
    random_state : int, optional, default=None

    Attributes
    ----------
    trimmer_ : BaseTransform or None
    encoder_ : SequenceOneHotEncoder
    vae_ : object
    generator_ : AptaDiffGenerator
    gmm_ : sklearn.mixture.GaussianMixture
    gmm_centers_ : numpy.ndarray
    latents_ : numpy.ndarray
    sequences_ : pandas.DataFrame
    seq_len_ : int
    embed_size_ : int

    References
    ----------
    .. [1] Wang, Z., et al. "AptaDiff: de novo design and optimization of
           aptamers based on diffusion models." Briefings in Bioinformatics,
           25(6), bbae517 (2024). https://doi.org/10.1093/bib/bbae517

           Original implementation: https://github.com/wz-create/AptaDiff
    """

    def __init__(
        self,
        vae,
        trimmer=None,
        n_clusters: int = 8,
        gmm_restarts: int = 100,
        generator=None,
        random_state: int | None = None,
    ) -> None:
        self.vae = vae
        self.trimmer = trimmer
        self.n_clusters = n_clusters
        self.gmm_restarts = gmm_restarts
        self.generator = generator
        self.random_state = random_state

    def fit(self, X: ArrayLike, y=None) -> "AptaDiff":
        """Encode the reads, extract their latents, and train the model.

        Parameters
        ----------
        X : MoleculeLoader or pandas.DataFrame
            Raw SELEX reads, one read per row, in a single column.
        y : ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        AptaDiff
            The fitted estimator.
        """
        pass

    def sample(self, n_samples: int = 1) -> pd.DataFrame:
        """Generate new aptamer sequences.

        Parameters
        ----------
        n_samples : int, optional, default=1
            Number of sequences to generate.

        Returns
        -------
        pandas.DataFrame
            Generated sequences in a column named ``"sequence"``.
        """
        pass

    def transform(self, X: ArrayLike) -> np.ndarray:
        """Encode sequences to their latent coordinates.

        Parameters
        ----------
        X : MoleculeLoader or pandas.DataFrame
            Raw reads, in the same form as passed to `fit`.

        Returns
        -------
        numpy.ndarray
            Latent coordinates, shape (n_samples, embed_size_).
        """
        pass

    def inverse_transform(self, Z: ArrayLike) -> pd.DataFrame:
        """Generate one sequence per latent point.

        Parameters
        ----------
        Z : array-like
            Latent coordinates, shape (n_points, embed_size_).

        Returns
        -------
        pandas.DataFrame
            Generated sequences in a column named ``"sequence"``.
        """
        pass

    def reconstruct(self, X: ArrayLike) -> pd.DataFrame:
        """Encode sequences and regenerate them from their own latents.

        Parameters
        ----------
        X : MoleculeLoader or pandas.DataFrame
            Raw reads, in the same form as passed to `fit`.

        Returns
        -------
        pandas.DataFrame
            Reconstructed sequences in a column named ``"sequence"``.
        """
        pass

    def score(self, X: ArrayLike, y=None) -> float:
        """Average log-likelihood of the sequences, in bits per token.

        Parameters
        ----------
        X : MoleculeLoader or pandas.DataFrame
            Raw reads, in the same form as passed to `fit`.
        y : ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        float
            Mean of `score_samples`.
        """
        raise NotImplementedError

    def score_samples(self, X: ArrayLike) -> np.ndarray:
        """Per-sequence log-likelihood, in bits per token.

        Parameters
        ----------
        X : MoleculeLoader or pandas.DataFrame
            Raw reads, in the same form as passed to `fit`.

        Returns
        -------
        numpy.ndarray
            One value per sequence, shape (n_samples,).
        """
        pass

    def _latents(self, X_idx: torch.Tensor) -> np.ndarray:
        """Extract latent coordinates from `vae`.

        Parameters
        ----------
        X_idx : torch.Tensor
            Integer-encoded sequences, shape (n_samples, seq_len), dtype
            ``torch.long``.

        Returns
        -------
        numpy.ndarray
            Latent coordinates, shape (n_samples, embed_size_).

        Raises
        ------
        TypeError
            If `vae` exposes neither ``transform`` nor ``encoder`` and
            ``h2mu``.
        """
        pass

    def _prepare(
        self, X: ArrayLike, fit: bool = False
    ) -> tuple[pd.DataFrame, torch.Tensor, pd.DataFrame]:
        """Clean and encode raw reads into numeric representations.

        Parameters
        ----------
        X : MoleculeLoader or pandas.DataFrame
            Raw reads, one read per row, in a single column.
        fit : bool, optional, default=False
            Whether to fit `trimmer` and the one-hot encoder, or to reuse the
            ones stored by `fit`.

        Returns
        -------
        sequences : pandas.DataFrame
            The cleaned sequences that survived encoding.
        X_idx : torch.Tensor
            Integer-encoded sequences, shape (n_samples, seq_len), dtype
            ``torch.long``.
        Y : pandas.DataFrame
            One-hot encoded sequences, shape (n_samples, num_classes *
            seq_len), as :class:`~pyaptamer.trafos.encode.SequenceOneHotEncoder`
            produces them.
        """
        raise NotImplementedError

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.non_deterministic = True
        return tags
