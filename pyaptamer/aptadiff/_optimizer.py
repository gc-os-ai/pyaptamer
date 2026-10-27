"""Bayesian optimization over the AptaDiff latent space for wet-lab candidates."""

__author__ = ["aditi-dsi"]
__all__ = ["AptaDiffOptimizer"]

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from skbase.base import BaseObject


class AptaDiffOptimizer(BaseObject):
    """Propose the next batch of aptamer candidates for wet-lab evaluation [1]_.


    References
    ----------
    .. [1] Wang, Z., et al. "AptaDiff: de novo design and optimization of
           aptamers based on diffusion models." Briefings in Bioinformatics,
           25(6), bbae517 (2024). https://doi.org/10.1093/bib/bbae517

           Original implementation: https://github.com/wz-create/AptaDiff
    """

    def __init__(
        self,
        aptadiff,
        bounds=(-4, 4),
        batch_size: int = 10,
        kappa: float = 2.0,
        random_state=None,
        encode_sequences: bool = False,
    ) -> None:
        self.aptadiff = aptadiff
        self.bounds = bounds
        self.batch_size = batch_size
        self.kappa = kappa
        self.random_state = random_state
        self.encode_sequences = encode_sequences
        super().__init__()

    def initial_design(self) -> pd.DataFrame:
        """Decode the cluster centers into the first batch of candidates.

        Returns
        -------
        pandas.DataFrame
            One row per cluster center, in the layout described in `propose`.
            The ``drift`` column is filled, the activities are not yet known.
        """
        pass

    def propose(self, X: ArrayLike, y: ArrayLike) -> pd.DataFrame:
        """Propose the next batch of candidates from the measured ones.

        Parameters
        ----------
        X : array-like
            The candidates measured so far. Latent coordinates of shape
            (n_measured, embed_size), or raw reads if `encode_sequences` is
            True.
        y : array-like
            Measured binding activity of each candidate in `X`, shape
            (n_measured,). Higher is better; the sign is handled internally.

        Returns
        -------
        pandas.DataFrame
            One row per proposed candidate, with the generated ``sequence``,
            the proposed coordinates, the coordinates the sequence re-embeds
            to, and the ``drift`` between them.

        Raises
        ------
        NotFittedError
            If `aptadiff` has not been fitted.
        ValueError
            If `X` and `y` disagree in length, if `X` does not match the
            latent dimension of `aptadiff`, or if any coordinate in `X` falls
            outside `bounds`.
        """
        pass

    def _resolve_bounds(self, n_dims: int) -> np.ndarray:
        """Return `bounds` as an array of shape (n_dims, 2).

        Parameters
        ----------
        n_dims : int
            Number of latent dimensions the bounds must cover.

        Returns
        -------
        numpy.ndarray
            The resolved box, shape (n_dims, 2).
        """
        pass

    def _validate_inputs(self, X: ArrayLike, y: ArrayLike) -> tuple:
        """Check the measured candidates against `bounds` and `aptadiff`.

        Parameters
        ----------
        X : array-like
            Measured candidates, as passed to `propose`.
        y : array-like
            Measured activities, as passed to `propose`.

        Returns
        -------
        tuple of numpy.ndarray
            The validated coordinates and activities, ``(X, y)``. Coordinates
            outside `bounds` are reported by row, since `BayesianOptimizer`
            rejects them without saying which.
        """
        pass

    def _compute_drift(self, Z: np.ndarray, sequences: pd.DataFrame) -> np.ndarray:
        """Distance between proposed coordinates and where sequences re-embed.

        Parameters
        ----------
        Z : numpy.ndarray
            Proposed coordinates, shape (n_candidates, embed_size).
        sequences : pandas.DataFrame
            The sequences decoded from `Z`, in a column named ``"sequence"``.

        Returns
        -------
        numpy.ndarray
            Euclidean distance per candidate, shape (n_candidates,). Large
            values mark coordinates the model cannot realise as a sequence.
        """
        pass
