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
from sklearn.utils import Tags, check_random_state
from sklearn.utils.validation import check_is_fitted

from pyaptamer.aptadiff._generator import AptaDiffGenerator
from pyaptamer.data import MoleculeLoader
from pyaptamer.trafos.encode import SequenceOneHotEncoder


class AptaDiff(BaseEstimator):
    """Design and optimise aptamers with AptaDiff [1]_.

    Takes a pretrained latent model and AptaDiff's multinomial diffusion
    model. `fit` cleans and encodes raw SELEX reads, reads their latent
    coordinates out of `vae`, trains the diffusion model to generate sequences
    conditioned on those coordinates, and fits a Gaussian mixture to the
    latents. `sample` then generates new sequences from that mixture.

    There is no `predict`. AptaDiff is unsupervised and generative, so
    as per scikit-learn conventions, `sample` draws new sequences.
    Regenerating input sequences from their own latents is available as `reconstruct`.

    `score` returns the average log-likelihood in bits per character, negated so
    that higher is better as per scikit-learn conventions. The original implementation
    reports the same quantity unnegated.

    This class holds no training code of its own. `vae` is used directly for inference,
    and diffusion training is delegated to `generator`.

    Optimising aptamers is external. The diffusion model is
    conditioned only on latent coordinates, so steering generation
    towards better binders means choosing better latent points. `gmm_centers_`
    gives the motif-dependent modes used for de novo design, and
    `inverse_transform` accepts any latent points, including ones proposed by
    an outer loop such as Bayesian optimisation using wet-lab results.

    Parameters
    ----------
    vae : object
        Pretrained latent model mapping integer-encoded sequences to latent
        coordinates. Requires a trained model as an untrained model produces random,
        unusable coordinates. Accepted either as an object with a
        ``transform(X_idx)`` method returning shape (n_samples, embed_size), or
        as a module with ``encoder`` and ``h2mu`` attributes such as
        :class:`~pyaptamer.raptgen.CNNPHMMVAE`. Anything else raises
        ``TypeError``. This class never trains the model, and never inspects or
        migrates its weights.
    trimmer : BaseTransform or None, optional, default=None
        Transform applied to the raw reads before encoding, typically
        :class:`~pyaptamer.trafos.transform.PrimerTrimmer` to strip the
        constant primer regions of a SELEX read. If None, `X` must already
        hold sequences of one fixed length.
    encoder : SequenceOneHotEncoder or None, optional, default=None
        Transform that encodes the cleaned sequences in one-hot form, cloned before
        fitting. It is both the diffusion model's target and the decoder that
        turns generated output back into strings, so its vocabulary supports the
        encoding in both directions. If None, uses
        ``SequenceOneHotEncoder(handle_unknown="drop")``, which treats A/T/G/C
        as four classes with ``U`` sharing ``T``'s index and drops reads
        holding any other character, reproducing the original implementation's
        filtering of ambiguous chars. Inject a pre-configured
        :class:`~pyaptamer.trafos.encode.SequenceOneHotEncoder` to supply a
        custom `vocab` or `inverse_vocab`, or to make unknown characters raise
        rather than be dropped.
    n_clusters : int, optional, default=8
        Number of Gaussian mixture components fitted to the training latents.
        Must not exceed the number of valid sequences left after cleaning.
        The default matches the original implementation.
    gmm_restarts : int, optional, default=100
        Number of Gaussian mixture fits attempted, of which the one with the
        lowest AIC is kept. Must be at least 1. The default matches the
        original implementation.
    generator : AptaDiffGenerator or None, optional, default=None
        The diffusion engine, cloned before fitting. If None, uses
        :class:`~pyaptamer.aptadiff.AptaDiffGenerator` with its defaults,
        that matches the original paper's configuration, which runs one denoiser
        pass per diffusion timestep.
    random_state : int, optional, default=None
        Seed for the mixture fits, for drawing latent points in `sample`, and
        for the diffusion model when `generator` is None. The global NumPy and
        Torch random states are left unchanged.

    Attributes
    ----------
    trimmer_ : BaseTransform or None
        The fitted clone of `trimmer`, or None if none was given.
    encoder_ : SequenceOneHotEncoder
        The fitted clone of `encoder`, or of the default encoder if none was
        given. Supplies the diffusion model's target and decodes generated
        sequences back to strings.
    vae_ : object
        The latent model used for extraction. Its weights are never read,
        written or migrated by this class.
    generator_ : AptaDiffGenerator
        The fitted diffusion engine.
    gmm_ : sklearn.mixture.GaussianMixture
        The mixture fitted to `latents_`, selected by lowest AIC.
    gmm_centers_ : numpy.ndarray
        Component means of `gmm_`, shape (n_clusters, embed_size_). Passing
        these to `inverse_transform` reproduces the original implementation's
        exact sequence generation.
    latents_ : numpy.ndarray
        Latent coordinates of the training sequences, shape
        (n_samples, embed_size_).
    sequences_ : pandas.DataFrame
        The valid sequences left after cleaning. Each row matches the
        corresponding coordinate in `latents_`.
    seq_len_ : int
        Length of the training sequences, and of the sequences generated.
    embed_size_ : int
        Dimension of the latent coordinates, inferred from the output of `vae`.


    References
    ----------
    .. [1] Wang, Z., et al. "AptaDiff: de novo design and optimization of
           aptamers based on diffusion models." Briefings in Bioinformatics,
           25(6), bbae517 (2024). https://doi.org/10.1093/bib/bbae517

           Original implementation: https://github.com/wz-create/AptaDiff

    Examples
    --------
    >>> import torch
    >>> from pyaptamer.aptadiff import AptaDiff, AptaDiffGenerator
    >>> from pyaptamer.datasets import load_sample_fastq
    >>> from pyaptamer.raptgen import CNNPHMMVAE
    >>> from pyaptamer.trafos.transform import PrimerTrimmer
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

    Generate new sequences, or condition on the motif modes to reproduce the
    paper's exact sequence generation:

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
        """Fit the mixture several times and keep the best fit.

        Each restart is initialised differently but the sequence of restarts is
        determined by `random_state`, so the selected fit is reproducible. The
        fit with the lowest AIC is kept, as in the original implementation.

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
        random_state = check_random_state(self.random_state)
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
        """Generate new aptamer sequences.

        Draws latent points from `gmm_` and runs the reverse diffusion process
        conditioned on each point.

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

        Notes
        -----
        The original paper generates exactly `n_clusters` sequences by strictly
        using the cluster centers. This method is more flexible, allowing
        to generate any `n_samples` by randomly sampling around those centers.
        To reproduce the paper's exact generation output, pass `gmm_centers_`
        to `inverse_transform` instead.
        """
        check_is_fitted(self)

        random_state = self.gmm_.random_state
        self.gmm_.random_state = check_random_state(self.random_state)
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

        Accepts any latent points, including ones chosen outside this
        estimator, such as the output of an optimisation loop over wet-lab
        affinity measurements.

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
            Reconstructed sequences in a column named ``"sequence"``, indexed
            by the rows of `X` they were reconstructed from, so they can be
            compared against the input row by row.

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
        """Average log-likelihood of the sequences, in bits per character.

        It is negated so that higher is better, as per scikit-learn conventions.
        The original implementation reports the same quantity unnegated, as bits
        per character.

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
        """Per-sequence log-likelihood, in bits per character.

        It is negated so that higher is better, as per scikit-learn conventions.

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
            seq_len), as :class:`~pyaptamer.trafos.encode.SequenceOneHotEncoder`
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
        else:
            reads = X.iloc[:, 0]
            if pd.api.types.infer_dtype(reads, skipna=True) == "string":
                lengths = reads.dropna().str.len()
                if lengths.nunique() > 1:
                    raise ValueError(
                        f"{type(self).__name__} received sequences of variable "
                        f"lengths {sorted(lengths.unique())}, but the diffusion "
                        "model generates one fixed length. Clean the reads "
                        "beforehand, or pass an instantiated PrimerTrimmer as "
                        "the trimmer argument to strip the constant primer "
                        "regions and keep the fixed-length random region."
                    )

        Y = self.encoder_.fit_transform(X) if fit else self.encoder_.transform(X)
        if len(Y) == 0:
            raise ValueError(
                f"{type(self).__name__} has no sequences left to work with, "
                "every read was dropped while cleaning and encoding."
            )

        sequences = X.loc[Y.index]
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

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.non_deterministic = True
        return tags
