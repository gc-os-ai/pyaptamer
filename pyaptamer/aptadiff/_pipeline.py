"""End-to-end AptaDiff algorithm taking SELEX reads and generating aptamer sequences."""

__author__ = ["aditi-dsi"]
__all__ = ["AptaDiff"]

import math

import numpy as np
import pandas as pd
import torch
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, clone
from sklearn.mixture import GaussianMixture
from sklearn.utils.validation import check_is_fitted

from pyaptamer.aptadiff._generator import AptaDiffGenerator
from pyaptamer.data import MoleculeLoader
from pyaptamer.transformations.encode import SequenceOneHotEncoder


class AptaDiff(BaseEstimator):
    """Generate aptamer sequences with AptaDiff [1]_.

    `fit` takes raw SELEX reads. It trims and one-hot encodes them, reads
    their latent coordinates from the pretrained `vae`, trains the diffusion
    model of `generator` to produce each encoded read from its coordinates,
    and fits a Gaussian mixture to the coordinates. `sample` draws
    coordinates from the mixture and decodes them into sequences.
    `inverse_transform` decodes any coordinates, for example the cluster
    centers in `gmm_centers_` or points proposed by an outer optimization
    loop.

    Parameters
    ----------
    vae : object
        Pretrained latent model. Either an object with a ``transform(X_idx)``
        method that maps integer-encoded sequences to an array of shape
        (n_samples, embed_size), or a module with ``encoder`` and ``h2mu``
        attributes such as :class:`~pyaptamer.raptgen.CNNPHMMVAE`. The model
        is only used for inference. It must be trained already.
    trimmer : BaseTransform or None, optional, default=None
        Transform applied to the reads before encoding, typically
        :class:`~pyaptamer.transformations.transform.PrimerTrimmer`. If None,
        the reads must all have the same length.
    encoder : SequenceOneHotEncoder or None, optional, default=None
        One-hot encoder for the trimmed reads. Its vocabulary also gives the
        integer indices passed to `vae`, so it must match the vocabulary the
        latent model was trained with. If None, uses
        ``SequenceOneHotEncoder(handle_unknown="drop")``: A, T, G and C are
        the four classes, U shares the index of T, and reads with any other
        character are dropped. The default matches
        :class:`~pyaptamer.raptgen.CNNPHMMVAE`.
    n_clusters : int, optional, default=8
        Number of components of the Gaussian mixture. Must not exceed the
        number of reads left after trimming and encoding.
    gmm_restarts : int, optional, default=100
        Number of mixture fits. The fit with the lowest AIC is kept. Must be
        at least 1.
    generator : AptaDiffGenerator or None, optional, default=None
        The diffusion model. If None, uses
        :class:`~pyaptamer.aptadiff.AptaDiffGenerator` with its defaults.
    random_state : int or None, optional, default=None
        Seed for the mixture fits, for `sample`, and for `generator` when it
        is None. The global NumPy random state is never used. The global
        Torch random state is left unchanged when a seed is given.

    Attributes
    ----------
    trimmer_ : BaseTransform or None
        Fitted clone of `trimmer`, or None.
    encoder_ : SequenceOneHotEncoder
        Fitted clone of `encoder`, or the fitted default encoder.
    vae_ : object
        The `vae` that was passed, unchanged.
    generator_ : AptaDiffGenerator
        Fitted clone of `generator`, or the fitted default generator.
    gmm_ : sklearn.mixture.GaussianMixture
        The mixture with the lowest AIC over `gmm_restarts` fits.
    gmm_centers_ : numpy.ndarray
        Means of `gmm_`, shape (n_clusters, embed_size_).
    latents_ : numpy.ndarray
        Latent coordinates of the training reads, shape
        (n_samples, embed_size_).
    sequences_ : pandas.DataFrame
        The training reads left after trimming and encoding, one row per row
        of `latents_`.
    seq_len_ : int
        Length of the encoded reads and of the generated sequences.
    embed_size_ : int
        Number of latent coordinates per read.

    Notes
    -----
    The original implementation generates one sequence per cluster center.
    ``inverse_transform(gmm_centers_)`` does the same. `sample` draws new
    points from the mixture instead.

    `score` and `score_samples` return the log-likelihood in bits per
    character, so higher is better. The original implementation reports the
    negative of this value as a loss.

    References
    ----------
    .. [1] Wang, Z., et al. "AptaDiff: de novo design and optimization of
           aptamers based on diffusion models." Briefings in Bioinformatics,
           25(6), bbae517 (2024). https://doi.org/10.1093/bib/bbae517

           Original implementation: https://github.com/wz-create/AptaDiff

    Examples
    --------
    The example uses an untrained latent model and a tiny diffusion model.

    >>> import torch
    >>> from pyaptamer.aptadiff import AptaDiff, AptaDiffGenerator
    >>> from pyaptamer.datasets import load_sample_fastq
    >>> from pyaptamer.raptgen import CNNPHMMVAE
    >>> from pyaptamer.transformations.transform import PrimerTrimmer
    >>>
    >>> vae = CNNPHMMVAE(motif_len=40, embed_size=2)
    >>>
    >>> trimmer = PrimerTrimmer(
    ...     start_primer="TAATACGACTCACTATAGGGAGAACTTCGACCAGAAG",
    ...     end_primer="TATGTGCGCATACATGGATCCTC",
    ...     variable_length=40,
    ... )
    >>>
    >>> generator = AptaDiffGenerator(
    ...     num_timesteps=5, dim=32, depth=1, max_epochs=1, random_state=0
    ... )
    >>>
    >>> aptadiff = AptaDiff(
    ...     vae=vae,
    ...     trimmer=trimmer,
    ...     generator=generator,
    ...     n_clusters=3,
    ...     gmm_restarts=5,
    ...     random_state=0,
    ... )
    >>> aptadiff.fit(load_sample_fastq())  # doctest: +ELLIPSIS
    AptaDiff(...)

    Generate new sequences, or use the learned cluster centers to reproduce
    the paper's exact sequence generation:

    >>> aptadiff.sample(2)["sequence"].str.len().tolist()
    [40, 40]
    >>> aptadiff.inverse_transform(aptadiff.gmm_centers_).shape
    (3, 1)
    """

    def __init__(
        self,
        vae,
        trimmer=None,
        encoder=None,
        n_clusters: int = 8,
        gmm_restarts: int = 100,
        generator=None,
        random_state: int | None = None,
    ) -> None:
        self.vae = vae
        self.trimmer = trimmer
        self.encoder = encoder
        self.n_clusters = n_clusters
        self.gmm_restarts = gmm_restarts
        self.generator = generator
        self.random_state = random_state

    def fit(self, X: ArrayLike, y=None) -> "AptaDiff":
        """Trim and encode the reads, train the diffusion model, fit the mixture.

        Parameters
        ----------
        X : MoleculeLoader or pandas.DataFrame
            Raw SELEX reads, one read per row, in the first column.
        y : ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        AptaDiff
            The fitted estimator.

        Raises
        ------
        ValueError
            If `gmm_restarts` is less than 1, if no sequence survives cleaning
            and encoding, if the number of classes the encoder produces does
            not match the generator's `num_classes`, or if fewer sequences
            survive than `n_clusters`.
        """
        if self.gmm_restarts < 1:
            raise ValueError(
                f"gmm_restarts must be at least 1, got {self.gmm_restarts}."
            )

        sequences, X_idx, Y = self._prepare(X, fit=True)
        latents = self._latents(X_idx)

        self.seq_len_ = X_idx.shape[1]
        self.embed_size_ = latents.shape[1]

        generator = self.generator
        if generator is None:
            generator = AptaDiffGenerator(random_state=self.random_state)

        num_classes = Y.shape[1] // self.seq_len_
        expected = getattr(generator, "num_classes", None)
        if expected is not None and expected != num_classes:
            raise ValueError(
                f"The encoder produces {num_classes} classes per position, but "
                f"the generator expects num_classes={expected}. Pass "
                f"AptaDiffGenerator(num_classes={num_classes}) as the generator, "
                f"or an encoder whose vocabulary has {expected} classes."
            )

        self.generator_ = clone(generator).fit(latents, Y.to_numpy())
        self.gmm_ = self._fit_gmm(latents)
        self.vae_ = self.vae
        self.latents_ = latents
        self.sequences_ = sequences
        self.gmm_centers_ = self.gmm_.means_

        return self

    def _fit_gmm(self, latents: np.ndarray) -> GaussianMixture:
        """Fit the mixture `gmm_restarts` times and keep the fit with the lowest AIC.

        Parameters
        ----------
        latents : numpy.ndarray
            Latent coordinates of the training sequences, shape
            (n_samples, embed_size_).

        Returns
        -------
        sklearn.mixture.GaussianMixture
            The fit with the lowest AIC out of `gmm_restarts` attempts.
        """
        random_state = np.random.RandomState(self.random_state)
        best_gmm, best_aic = None, np.inf

        for _ in range(self.gmm_restarts):
            gmm = GaussianMixture(
                self.n_clusters, covariance_type="full", random_state=random_state
            ).fit(latents)
            aic = gmm.aic(latents)
            if aic < best_aic:
                best_gmm, best_aic = gmm, aic

        return best_gmm

    def sample(self, n_samples: int = 1) -> pd.DataFrame:
        """Generate new sequences.

        Draws latent points from `gmm_` and decodes them with `inverse_transform`.

        Parameters
        ----------
        n_samples : int, optional, default=1
            Number of sequences to generate.

        Returns
        -------
        pandas.DataFrame
            Generated sequences in a column named ``"sequence"``.

        Raises
        ------
        NotFittedError
            If `fit` has not been called before.
        """
        check_is_fitted(self)

        random_state = self.gmm_.random_state
        self.gmm_.random_state = np.random.RandomState(self.random_state)
        try:
            latents, _ = self.gmm_.sample(n_samples)
        finally:
            self.gmm_.random_state = random_state

        return self.inverse_transform(latents)

    def transform(self, X: ArrayLike) -> np.ndarray:
        """Encode sequences to their latent coordinates.

        Parameters
        ----------
        X : MoleculeLoader or pandas.DataFrame
            Raw reads, in the same form as passed to `fit`.

        Returns
        -------
        numpy.ndarray
            Latent coordinates, shape (n_samples, embed_size_). Reads dropped
            while cleaning and encoding are not represented.

        Raises
        ------
        NotFittedError
            If `fit` has not been called before.
        ValueError
            If no sequence survives cleaning and encoding, or if `X` holds
            sequences of a length other than `seq_len_`.
        """
        check_is_fitted(self)
        _, X_idx, _ = self._prepare(X, fit=False)
        return self._latents(X_idx)

    def inverse_transform(self, Z: ArrayLike) -> pd.DataFrame:
        """Generate one sequence per latent point.

        The points can come from anywhere, for example `gmm_centers_` or an
        optimization loop.

        Parameters
        ----------
        Z : array-like
            Latent coordinates, shape (n_points, embed_size_).

        Returns
        -------
        pandas.DataFrame
            Generated sequences in a column named ``"sequence"``.

        Raises
        ------
        NotFittedError
            If `fit` has not been called before.
        ValueError
            If `Z` does not have `embed_size_` columns.
        """
        check_is_fitted(self)
        onehot = self.generator_.predict(Z)
        return self.encoder_.inverse_transform(pd.DataFrame(onehot))

    def reconstruct(self, X: ArrayLike) -> pd.DataFrame:
        """Encode sequences and regenerate them from their own latents.

        Parameters
        ----------
        X : MoleculeLoader or pandas.DataFrame
            Raw reads, in the same form as passed to `fit`.

        Returns
        -------
        pandas.DataFrame
            Reconstructed sequences in a column named ``"sequence"``, with the
            index of the rows of `X` they come from.

        Raises
        ------
        NotFittedError
            If `fit` has not been called before.
        ValueError
            If no sequence survives cleaning and encoding, or if `X` holds
            sequences of a length other than `seq_len_`.
        """
        check_is_fitted(self)
        sequences, X_idx, _ = self._prepare(X, fit=False)
        reconstructed = self.inverse_transform(self._latents(X_idx))
        reconstructed.index = sequences.index
        return reconstructed

    def score(self, X: ArrayLike, y=None) -> float:
        """Mean log-likelihood of the reads in bits per character.

        Higher is better. The original implementation reports the negative of
        this value as a loss.

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

        Raises
        ------
        NotFittedError
            If `fit` has not been called before.
        ValueError
            If no sequence survives cleaning and encoding, or if `X` holds
            sequences of a length other than `seq_len_`.
        """
        check_is_fitted(self)
        return float(self.score_samples(X).mean())

    def score_samples(self, X: ArrayLike) -> np.ndarray:
        """Log-likelihood of each read in bits per character.

        Higher is better. The original implementation reports the negative of
        this value as a loss.

        Parameters
        ----------
        X : MoleculeLoader or pandas.DataFrame
            Raw reads, in the same form as passed to `fit`.

        Returns
        -------
        numpy.ndarray
            One value per sequence, shape (n_samples,). Reads dropped while
            cleaning and encoding are not represented.

        Raises
        ------
        NotFittedError
            If `fit` has not been called before.
        ValueError
            If no sequence survives cleaning and encoding, or if `X` holds
            sequences of a length other than `seq_len_`.
        """
        check_is_fitted(self)
        _, X_idx, Y = self._prepare(X, fit=False)
        latents = self._latents(X_idx)

        diffusion = self.generator_.diffusion_
        num_classes = diffusion.num_classes
        device = next(diffusion.parameters()).device

        x = torch.as_tensor(Y.to_numpy().copy())
        x = x.reshape(len(Y), num_classes, self.seq_len_).to(device)
        z = torch.as_tensor(latents, dtype=x.dtype).to(device)

        was_training = diffusion.training
        diffusion.eval()
        try:
            with torch.random.fork_rng(enabled=self.random_state is not None):
                if self.random_state is not None:
                    torch.manual_seed(self.random_state)
                with torch.no_grad():
                    log_prob = diffusion.log_prob(x, z)
        finally:
            diffusion.train(was_training)

        bits_per_token = log_prob / (math.log(2) * self.seq_len_)
        return bits_per_token.cpu().numpy()

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
        vae = self.vae

        if hasattr(vae, "transform"):
            return np.asarray(vae.transform(X_idx))

        if hasattr(vae, "encoder") and hasattr(vae, "h2mu"):
            X_idx = X_idx.to(next(vae.parameters()).device)
            was_training = vae.training
            vae.eval()
            try:
                with torch.no_grad():
                    mu = vae.h2mu(vae.encoder(X_idx))
            finally:
                vae.train(was_training)
            return mu.cpu().numpy()

        raise TypeError(
            "vae must expose either a transform(X_idx) method returning an "
            "array of shape (n_samples, embed_size), or the attributes "
            "encoder and h2mu, as pyaptamer.raptgen.CNNPHMMVAE does. Got "
            f"{type(vae).__name__}, which has neither."
        )

    def _prepare(
        self, X: ArrayLike, fit: bool = False
    ) -> tuple[pd.DataFrame, torch.Tensor, pd.DataFrame]:
        """Clean and encode raw reads into numeric representations.

        Parameters
        ----------
        X : MoleculeLoader or pandas.DataFrame
            Raw reads, one read per row, in a single column.
        fit : bool, optional, default=False
            If True, set `trimmer_` and `encoder_` and fit them to `X`. If
            False, reuse the ones stored by `fit` and check that `X` holds
            sequences of the length that was fitted.

        Returns
        -------
        sequences : pandas.DataFrame
            The cleaned sequences that survived encoding.
        X_idx : torch.Tensor
            Integer-encoded sequences, shape (n_samples, seq_len), dtype
            ``torch.long``.
        Y : pandas.DataFrame
            One-hot encoded sequences, shape (n_samples, num_classes *
            seq_len), as
            :class:`~pyaptamer.transformations.encode.SequenceOneHotEncoder`
            produces them.

        Raises
        ------
        TypeError
            If `X` is neither a MoleculeLoader nor a pandas DataFrame.
        ValueError
            If no trimmer is set and `X` holds sequences of varying lengths,
            if no sequence survives cleaning and encoding, or if `fit` is
            False and `X` holds sequences of a length other than `seq_len_`.
        """
        X = X.to_dataframe() if isinstance(X, MoleculeLoader) else X
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "X must be a MoleculeLoader instance or a pandas DataFrame. "
                f"Got {type(X)} instead."
            )

        if fit:
            self.trimmer_ = None if self.trimmer is None else clone(self.trimmer)
            if self.encoder is None:
                self.encoder_ = SequenceOneHotEncoder(handle_unknown="drop")
            else:
                self.encoder_ = clone(self.encoder)

        if self.trimmer_ is not None:
            X = self.trimmer_.fit_transform(X) if fit else self.trimmer_.transform(X)

        reads = X.reset_index(drop=True)
        Y = (
            self.encoder_.fit_transform(reads)
            if fit
            else self.encoder_.transform(reads)
        )
        if len(Y) == 0:
            raise ValueError(
                f"{type(self).__name__} has no sequences left to work with, "
                "every read was dropped while cleaning and encoding."
            )

        sequences = X.iloc[Y.index]
        Y.index = sequences.index
        seq_len = len(sequences.iloc[0, 0])
        if not fit and seq_len != self.seq_len_:
            raise ValueError(
                f"X holds sequences of length {seq_len}, but this "
                f"{type(self).__name__} was fitted on length {self.seq_len_}."
            )

        num_classes = Y.shape[1] // seq_len
        onehot = torch.as_tensor(Y.to_numpy().copy())
        X_idx = onehot.reshape(len(Y), num_classes, seq_len).argmax(dim=1)
        return sequences, X_idx, Y
