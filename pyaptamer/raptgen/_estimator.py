"""RaptGen estimator on integer-encoded sequences"""

__author__ = ["NoorMajdoub"]
__all__ = ["RaptGenModel"]

import itertools
import tempfile

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from torch.utils.data import DataLoader, TensorDataset

from pyaptamer.raptgen._model import CNNPHMMVAE, CNNPHMMVAEFast
from pyaptamer.raptgen._train_lightning import RaptGenLightning
from pyaptamer.raptgen.layers._sampler import ProfileHMMSampler


class RaptGenModel(BaseEstimator, TransformerMixin):
    """
    RaptGen algorithm for unsupervised aptamer sequence generation.

    Wraps `CNNPHMMVAE` (or its faster variant, `CNNPHMMVAEFast`) in a
    sklearn-style estimator. Training runs through a Lightning `Trainer`
    with the steps defined in `RaptGenLightning`.
    Generation works by taking a point in latent space,
    decoding it into profile HMM transition and emission
    probabilities, then sampling a concrete A/T/G/C sequence from those
    probabilities using `ProfileHMMSampler`.

    Parameters
    ----------
    motif_len : int, optional, default=12
        Length of the profile HMM template the decoder reconstructs against.
    embed_size : int, optional, default=10
        Dimensionality of the latent space.
    hidden_size : int, optional, default=32
        Size of the shared hidden representation between encoder and decoder.
    kernel_size : int, optional, default=7
        Convolution kernel size used by the CNN encoder. Must be odd.
    fast : bool, optional, default=False
        If True, use `CNNPHMMVAEFast` (faster decoder/loss) instead of
        `CNNPHMMVAE`.
    max_epochs : int, optional, default=1000
        Maximum number of training epochs.
    patience : int or None, optional, default=50
        Stop training after this many epochs without improvement of the
        validation loss. None disables early stopping. The KL weight ramp
        of `beta_schedule` also runs over this many epochs, as in the
        original implementation.
    batch_size : int, optional, default=64
        Minibatch size used during training.
    lr : float, optional, default=1e-3
        Learning rate.
    optimizer : callable, optional, default=torch.optim.Adam
        Optimizer class, or any callable returning an optimizer when called
        with the model parameters and `lr`.
    optimizer_kwargs : dict, optional, default=None
        Extra keyword arguments passed to `optimizer`.
    validation_fraction : float, optional, default=0.1
        Fraction of `X` held out each `fit` call to drive early stopping.
    beta_schedule : bool, optional, default=True
        If True, ramp the KL weight linearly from 0 to `beta` over the first
        `patience` epochs.
    beta : float, optional, default=1.0
        Weight of the KL-divergence term in the VAE loss, used once past
        the annealing period (or throughout, if `beta_schedule=False`).
    force_matching : bool, optional, default=True
        If True, apply profile-HMM "force matching" regularization.
    force_epochs : int, optional, default=50
        Number of epochs to apply `force_matching` for.
    random_state : int or None, optional, default=None
        Seed for training and for sampling in `inverse_transform`.
    verbose : int, optional, default=0
        If non-zero, show the Lightning progress bar and model summary.

    Attributes
    ----------
    model_ : CNNPHMMVAE or CNNPHMMVAEFast
        The fitted VAE model, holding the best-validation-loss weights seen
        during training (not necessarily the final epoch's). Only present
        after calling `fit`.
    n_epochs_ : int
        Number of epochs run during `fit`.
    """

    def __init__(
        self,
        motif_len=12,
        embed_size=10,
        hidden_size=32,
        kernel_size=7,
        fast=False,
        max_epochs=1000,
        patience=50,
        batch_size=64,
        lr=1e-3,
        optimizer=torch.optim.Adam,
        optimizer_kwargs=None,
        validation_fraction=0.1,
        beta_schedule=True,
        beta=1.0,
        force_matching=True,
        force_epochs=50,
        random_state=None,
        verbose=0,
    ):
        self.motif_len = motif_len
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.fast = fast
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.lr = lr
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.validation_fraction = validation_fraction
        self.beta_schedule = beta_schedule
        self.beta = beta
        self.force_matching = force_matching
        self.force_epochs = force_epochs
        self.random_state = random_state
        self.verbose = verbose

    def _build_model(self):
        """Instantiate an untrained VAE matching this estimator's params."""
        model_cls = CNNPHMMVAEFast if self.fast else CNNPHMMVAE
        return model_cls(
            motif_len=self.motif_len,
            embed_size=self.embed_size,
            hidden_size=self.hidden_size,
            kernel_size=self.kernel_size,
        )

    def _train_val_split(self, n):
        """Split sample indices into train and validation sets."""
        if n < 2:
            raise ValueError(f"Need at least 2 sequences to fit, got {n}.")
        n_val = max(1, int(n * self.validation_fraction))
        perm = torch.randperm(n)
        return perm[n_val:], perm[:n_val]

    def fit(self, X_idx, y=None):
        """Train the VAE on a collection of aptamer sequences.

        Parameters
        ----------
        X_idx : array-like of shape (n_samples, seq_len)
            Integer-encoded sequences, as produced by `SequenceIndexEncoder`.
        y : ignored
            Present for sklearn API compatibility.

        Returns
        -------
        self : RaptGenModel
        """
        X_idx = torch.as_tensor(np.asarray(X_idx), dtype=torch.long)
        with torch.random.fork_rng(enabled=self.random_state is not None):
            if self.random_state is not None:
                torch.manual_seed(self.random_state)
            self._fit(X_idx)
        return self

    def _fit(self, X_idx):
        train_idx, val_idx = self._train_val_split(len(X_idx))
        train_loader = DataLoader(
            TensorDataset(X_idx[train_idx]), batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(X_idx[val_idx]), batch_size=self.batch_size
        )
        model = RaptGenLightning(
            self._build_model(),
            optimizer=self.optimizer,
            optimizer_kwargs=self.optimizer_kwargs,
            lr=self.lr,
            beta=self.beta,
            beta_schedule=self.beta_schedule,
            beta_epochs=self.patience or self.max_epochs,
            force_matching=self.force_matching,
            force_epochs=self.force_epochs,
        )

        with tempfile.TemporaryDirectory() as checkpoint_dir:
            checkpoint = ModelCheckpoint(
                dirpath=checkpoint_dir,
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                save_weights_only=True,
            )
            callbacks = [checkpoint]
            if self.patience is not None:
                callbacks.append(
                    EarlyStopping(
                        monitor="val_loss", mode="min", patience=self.patience
                    )
                )
            trainer = L.Trainer(
                max_epochs=self.max_epochs,
                accelerator="auto",
                devices=1,
                logger=False,
                callbacks=callbacks,
                enable_progress_bar=bool(self.verbose),
                enable_model_summary=bool(self.verbose),
            )
            trainer.fit(model, train_loader, val_loader)

            if checkpoint.best_model_path:
                best = torch.load(checkpoint.best_model_path, map_location="cpu")
                model.load_state_dict(best["state_dict"])

        self.model_ = model.model.to(trainer.strategy.root_device).eval()
        self.n_epochs_ = trainer.current_epoch

    def transform(self, X):
        """Encode sequences into latent-space points.

        Parameters
        ----------
        X : array-like of shape (n_samples, seq_len)
            Integer-encoded sequences, as produced by `SequenceIndexEncoder`.

        Returns
        -------
        ndarray of shape (n_samples, embed_size)
            Latent means of the input sequences.
        """
        check_is_fitted(self, "model_")
        self.model_.eval()

        device = next(self.model_.parameters()).device
        X_idx = torch.as_tensor(np.asarray(X), dtype=torch.long).to(device)
        with torch.no_grad():
            h = self.model_.encoder(X_idx)
            mu = self.model_.h2mu(h)
        return mu.cpu().numpy()

    def inverse_transform(self, Z, most_likely=True):
        """Generate full sequences from latent-space points.

        Parameters
        ----------
        Z : array of shape (n_points, embed_size)
            Latent-space points.
        most_likely : bool, optional, default=True
            If True, take the greedy most probable path through the profile HMM
            and fill insert positions with the best-scoring nucleotides. If
            False, draw one random path.

        Returns
        -------
        sequences : list of str
            One A/T/G/C sequence per latent-space point in `Z`.
        """
        check_is_fitted(self, "model_")
        self.model_.eval()

        device = next(self.model_.parameters()).device
        z = torch.as_tensor(np.asarray(Z), dtype=torch.float32).to(device)
        with torch.no_grad():
            transition_proba, emission_proba = self.model_.decoder(z)

        rng = np.random.default_rng(self.random_state)
        sequences = []
        for i in range(z.shape[0]):
            sampler = ProfileHMMSampler(
                transition_proba[i].cpu().numpy(),
                emission_proba[i].cpu().numpy(),
                proba_is_log=True,
                random_state=rng,
            )
            if most_likely:
                seq = self._decode_most_probable(sampler, rng)
            else:
                seq = sampler.sample(sequence_only=True).replace("_", "").upper()
            sequences.append(seq)
        return sequences

    def _decode_most_probable(self, sampler, rng, eval_max=256):
        """
        Turn most_probable()'s raw skeleton into a final, fully-decoded
        sequence.

        most_probable() leaves "_" for deletions and "N" for insertions
        (since the model has no letter preference at insert positions).
        This strips the deletions, tries every possible real nucleotide
        combination for the insert wildcards, scores each complete
        candidate with calc_seq_proba, and keeps the best-scoring one.

        Parameters
        ----------
        sampler : ProfileHMMSampler
            The sampler to generate and score candidates with.
        rng : numpy.random.Generator
            Generator used to subsample candidates above `eval_max`.
        eval_max : int, optional, default=256
            Number of possible generated candidates.

        Returns
        -------
        best_seq : str
            The highest-scoring fully-decoded candidate sequence.
        """
        skeleton = sampler.most_probable(sequence_only=True)
        pattern = skeleton.replace("_", "").replace("N", "*")

        n_wildcards = pattern.count("*")
        all_combos = list(itertools.product("ATGC", repeat=n_wildcards))

        if len(all_combos) > eval_max:
            idx = rng.choice(len(all_combos), size=eval_max, replace=False)
            all_combos = [all_combos[i] for i in idx]

        candidates = []
        for combo in all_combos:
            combo_iter = iter(combo)
            candidates.append(
                "".join(next(combo_iter) if ch == "*" else ch for ch in pattern)
            )

        scored = [(c, sampler.calc_seq_proba(c)) for c in candidates]
        best_seq, _ = max(scored, key=lambda x: x[1])
        return best_seq
