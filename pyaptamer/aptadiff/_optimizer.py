"""Bayesian optimization over the AptaDiff latent space for wet-lab candidates."""

__author__ = ["aditi-dsi"]
__all__ = ["AptaDiffOptimizer"]

import warnings

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from skbase.base import BaseObject
from sklearn.utils.validation import check_is_fitted

from pyaptamer.bayesian_optimization import BayesianOptimizer


class AptaDiffOptimizer(BaseObject):
    """Propose the next batch of aptamer candidates for wet-lab evaluation [1]_.

    Runs a round-based process over the latent space of a fitted
    :class:`~pyaptamer.aptadiff.AptaDiff`. `initial_design` decodes the cluster
    centers in ``gmm_centers_`` into a first batch, which is the batch the
    original implementation sent to the lab.
    Every later round calls `propose` with the coordinates of the candidates
    measured so far and their measured binding activities, which fits a
    Gaussian process over those coordinates and decodes the next batch.

    There is no in-silico scoring step. The objective is the lab measurement,
    so `propose` should be given the activities observed for the previous
    batch, as in the original implementation.

    Both methods return one table per round, holding the generated sequence,
    the coordinates it was decoded from, the coordinates it encodes back to,
    and the distance between the two called the `drift`. A candidate
    with a large drift sits at a coordinate that do not translate cleanly into
    realistic sequences, and hence, can be dropped before it reaches the output.

    Parameters
    ----------
    aptadiff : AptaDiff
        Fitted :class:`~pyaptamer.aptadiff.AptaDiff`, used for inference only
        and is not refitted. ``gmm_centers_`` seeds the process,
        ``inverse_transform`` decodes proposed coordinates, and ``vae`` encodes
        the decoded sequences back for the drift.
    bounds : tuple or list of tuple, optional, default=(-4, 4)
        Search range of the latent coordinates. Either ``(low, high)`` applied
        to every dimension, or one ``(low, high)`` pair per dimension. The
        default is the domain of the original AptaDiff implementation.
    batch_size : int, optional, default=10
        Number of candidates that `propose` returns per round. Must be >= 1.
    kappa : float, optional, default=2.0
        Exploration weight for choosing next points. Higher values push the
        optimizer to explore untested regions of the latent space.
    random_state : int, numpy.random.Generator, or None, optional, default=None
        Seed for the optimizer. It does not seed the decoding,
        which is seeded by the generator inside `aptadiff`.
    encode_sequences : bool, optional, default=False
        How `propose` reads its `X`. If False, `X` already holds latent
        coordinates, as in the original implementation, which reads the stored
        coordinates of the measured candidates. If True, `X` holds raw reads
        and is encoded with ``aptadiff.transform``, as the RaptGen driver
        does. Re-encoding a read does not always return the coordinate it was
        decoded from, so the two ways can disagree.

    Attributes
    ----------
    next_points_ : numpy.ndarray
        Coordinates proposed by the last `propose` call, shape
        (batch_size, embed_size).
    proposals_ : pandas.DataFrame
        Table returned by the last `initial_design` or `propose` call.
    drift_ : numpy.ndarray
        Drift of each candidate in `proposals_`, shape (n_candidates,).


    References
    ----------
    .. [1] Wang, Z., et al. "AptaDiff: de novo design and optimization of
           aptamers based on diffusion models." Briefings in Bioinformatics,
           25(6), bbae517 (2024). https://doi.org/10.1093/bib/bbae517

           Original implementation: https://github.com/wz-create/AptaDiff

    Examples
    --------
    >>> import numpy as np
    >>> import torch
    >>> from pyaptamer.aptadiff import (
    ...     AptaDiff,
    ...     AptaDiffGenerator,
    ...     AptaDiffOptimizer,
    ... )
    >>> from pyaptamer.datasets import load_sample_fastq
    >>> from pyaptamer.raptgen import CNNPHMMVAE
    >>> from pyaptamer.transformations.transform import PrimerTrimmer
    >>>
    >>> trimmer = PrimerTrimmer(
    ...     start_primer="TAATACGACTCACTATAGGGAGAACTTCGACCAGAAG",
    ...     end_primer="TATGTGCGCATACATGGATCCTC",
    ...     variable_length=40,
    ... )
    >>> generator = AptaDiffGenerator(
    ...     num_timesteps=5, dim=32, depth=1, max_epochs=1, random_state=0
    ... )
    >>> _ = torch.manual_seed(0)
    >>> aptadiff = AptaDiff(
    ...     vae=CNNPHMMVAE(motif_len=40, embed_size=2),
    ...     trimmer=trimmer,
    ...     generator=generator,
    ...     n_clusters=3,
    ...     gmm_restarts=5,
    ...     random_state=0,
    ... )
    >>> aptadiff.fit(load_sample_fastq())  # doctest: +ELLIPSIS
    AptaDiff(...)

    The first round decodes the cluster centers:

    >>> optimizer = AptaDiffOptimizer(aptadiff, batch_size=4, random_state=0)
    >>> first = optimizer.initial_design()
    >>> first.columns.tolist()
    ['sequence', 'z0', 'z1', 'rz0', 'rz1', 'drift']

    Later rounds take the measured activities and propose the next batch:

    >>> activity = np.array([0.2, 0.5, 0.9])
    >>> nxt = optimizer.propose(aptadiff.gmm_centers_, activity)
    >>> len(nxt)
    4
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
            Initial candidate sequences and their coordinates. Because these
            candidates have not yet been tested in the lab, no activity scores
            are included.

        Raises
        ------
        NotFittedError
            If `aptadiff` has not been fitted.
        """
        check_is_fitted(self.aptadiff)
        centers = np.asarray(self.aptadiff.gmm_centers_, dtype=float)
        return self._build_proposals(centers)

    def propose(self, X: ArrayLike, y: ArrayLike) -> pd.DataFrame:
        """Propose the next batch of candidate sequences based on previous
        lab results.

        Parameters
        ----------
        X : array-like
            The candidates measured so far. Latent coordinates of shape
            (n_measured, embed_size), or raw reads if `encode_sequences` is
            True.
        y : array-like
            Measured binding activity of each candidate in `X`, shape
            (n_measured,). Higher is better. The activities are passed on
            with the optimizer set to maximise them, (equivalent to minimizing
            the negated term, as done in original implrmentation).

        Returns
        -------
        pandas.DataFrame
            A DataFrame where each row is a proposed candidate containing:
            - ``sequence``: The generated DNA sequence string.
            - ``z0``, ``z1``, ...: The proposed latent coordinates.
            - ``rz0``, ``rz1``, ...: The re-embedded latent coordinates.
            - ``drift``: The Euclidean distance between proposed and re-embedded
                         coordinates.

        Raises
        ------
        NotFittedError
            If `aptadiff` has not been fitted.
        ValueError
            If `X` and `y` disagree in length, if `X` does not match the
            latent dimension of `aptadiff`, or if any coordinate in `X` falls
            outside `bounds`.
        """
        check_is_fitted(self.aptadiff)
        X, y, bounds = self._validate_inputs(X, y)

        optimizer = BayesianOptimizer(
            bounds=bounds,
            batch_size=self.batch_size,
            minimize=False,
            kappa=self.kappa,
            random_state=self.random_state,
        )
        self.next_points_ = optimizer.suggest(X, y)

        return self._build_proposals(self.next_points_)

    def _build_proposals(self, Z: np.ndarray) -> pd.DataFrame:
        """Decode coordinates Z in one batch and format the results into
        a DataFrame.

        Parameters
        ----------
        Z : numpy.ndarray
            Coordinates to decode, shape (n_candidates, embed_size).

        Returns
        -------
        pandas.DataFrame
            The candidates, in the layout described in `propose`.
        """
        if getattr(self.aptadiff.generator_, "random_state", None) is None:
            warnings.warn(
                f"{type(self).__name__} is decoding with a generator that has "
                "random_state=None, so this round cannot be reproduced later. "
                "Pass an AptaDiffGenerator with a random_state to AptaDiff to "
                "get the same sequences back.",
                UserWarning,
                stacklevel=3,
            )

        sequences = self.aptadiff.inverse_transform(Z)
        reembedded, drift = self._compute_drift(Z, sequences)

        proposals = pd.DataFrame({"sequence": sequences["sequence"].to_numpy()})
        for i in range(Z.shape[1]):
            proposals[f"z{i}"] = Z[:, i]
        for i in range(reembedded.shape[1]):
            proposals[f"rz{i}"] = reembedded[:, i]
        proposals["drift"] = drift

        self.proposals_ = proposals
        self.drift_ = drift
        return proposals

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

        Raises
        ------
        ValueError
            If `bounds` covers a number of dimensions other than `n_dims`, or
            if any low is not less than high.
        """
        bounds = np.asarray(self.bounds, dtype=float)
        if bounds.shape == (2,):
            bounds = np.tile(bounds, (n_dims, 1))

        if bounds.shape != (n_dims, 2):
            raise ValueError(
                f"bounds must be a (low, high) pair or {n_dims} pairs, one per "
                f"latent dimension. Got shape {bounds.shape}."
            )
        if np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError("Each bound must satisfy low < high.")

        return bounds

    def _validate_inputs(
        self, X: ArrayLike, y: ArrayLike
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Check the measured candidates against `bounds` and `aptadiff`.

        Parameters
        ----------
        X : array-like
            Measured candidates, as passed to `propose`.
        y : array-like
            Measured activities, as passed to `propose`.

        Returns
        -------
        X : numpy.ndarray
            Latent coordinates of the measured candidates, shape
            (n_measured, embed_size).
        y : numpy.ndarray
            Measured activities, shape (n_measured,).
        bounds : numpy.ndarray
            The resolved search range, shape (embed_size, 2).

        Raises
        ------
        ValueError
            If `X` is not two-dimensional, if `X` and `y` disagree in length,
            if `X` does not match the latent dimension of `aptadiff`, or if
            any coordinate in `X` falls outside `bounds`.
        """
        if self.encode_sequences:
            X = self.aptadiff.transform(X)

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        if X.ndim != 2:
            raise ValueError(f"X must be two-dimensional, got shape {X.shape}.")
        if len(X) != len(y):
            raise ValueError(
                f"X and y must have the same number of rows, got {len(X)} and "
                f"{len(y)}. With encode_sequences=True the reads dropped while "
                "cleaning and encoding are not represented in X."
            )

        embed_size = self.aptadiff.embed_size_
        if X.shape[1] != embed_size:
            raise ValueError(
                f"X holds {X.shape[1]} coordinates per candidate, but aptadiff "
                f"was fitted with embed_size_={embed_size}."
            )

        bounds = self._resolve_bounds(embed_size)
        outside = np.flatnonzero(
            (X < bounds[:, 0]).any(axis=1) | (X > bounds[:, 1]).any(axis=1)
        )
        if len(outside) > 0:
            raise ValueError(
                f"X holds {len(outside)} candidate(s) outside bounds, at row(s) "
                f"{outside.tolist()}. Expand the bounds to cover them."
            )

        return X, y, bounds

    def _compute_drift(
        self, Z: np.ndarray, sequences: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Encode the decoded sequences back and measure how far they moved.

        The sequences are already the variable region, so the trimmer of
        `aptadiff` is skipped as it looks for the constant primer regions of a
        read and would drop every generated sequence.

        Parameters
        ----------
        Z : numpy.ndarray
            Coordinates the sequences were decoded from, shape
            (n_candidates, embed_size).
        sequences : pandas.DataFrame
            The sequences decoded from `Z`, in a column named ``"sequence"``.

        Returns
        -------
        reembedded : numpy.ndarray
            Coordinates the sequences encode back to, shape
            (n_candidates, embed_size).
        drift : numpy.ndarray
            Distance between proposed coordinates and their re-embedded coordinates.
            Large values highlight coordinates that do not translate cleanly into
            realistic sequences.

        Raises
        ------
        ValueError
            If a sequence is dropped while encoding, which would leave the
            coordinates and the sequences out of step.
        """
        trimmer = self.aptadiff.trimmer_
        self.aptadiff.trimmer_ = None
        try:
            reembedded = self.aptadiff.transform(sequences)
        finally:
            self.aptadiff.trimmer_ = trimmer

        if len(reembedded) != len(Z):
            raise ValueError(
                f"{len(Z) - len(reembedded)} of {len(Z)} generated sequence(s) "
                "were dropped during re-encoding, causing the coordinates and "
                "re-embedded results to become misaligned."
            )

        drift = np.linalg.norm(Z - reembedded, axis=1)
        return reembedded, drift
