"""RaptGen training/generation pipeline"""

__author__ = ["NoorMajdoub"]
__all__ = ["RaptGenModel"]

import itertools

import numpy as np
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from pyaptamer.raptgen._model import CNN_PHMM_VAE, CNN_PHMM_VAE_FAST
from pyaptamer.raptgen.layers._sampler import ProfileHMMSampler


class RaptGenModel(BaseEstimator, TransformerMixin):
    """
    RaptGen algorithm for unsupervised aptamer sequence generation.

    Wraps `CNN_PHMM_VAE` (or its faster variant, `CNN_PHMM_VAE_FAST`) in a
    sklearn-style estimator.
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
        If True, use `CNN_PHMM_VAE_FAST` (faster decoder/loss) instead of
        `CNN_PHMM_VAE`.
    epochs : int, optional, default=1000
        Maximum number of training epochs.
    batch_size : int, optional, default=64
        Minibatch size used during training.
    lr : float, optional, default=1e-3
        Learning rate for the Adam optimizer.
    validation_fraction : float, optional, default=0.1
        Fraction of `X` held out each `fit` call to drive early stopping.
    threshold : int, optional, default=50
        Early-stopping patience.
    beta_schedule : bool, optional, default=True.
    beta : float, optional, default=1.0
        Weight of the KL-divergence term in the VAE loss, used once past
        the annealing period (or throughout, if `beta_schedule=False`).
    force_matching : bool, optional, default=True
        If True, apply profile-HMM "force matching" regularization.
    force_epochs : int, optional, default=50
        Number of epochs to apply `force_matching` for.
    device : str or None, optional, default=None
        Torch device to train/run on.
    random_state : int or None, optional, default=None
        Seed for torch's RNG, for reproducible training/generation.

    Attributes
    ----------
    model_ : CNN_PHMM_VAE or CNN_PHMM_VAE_FAST
        The fitted VAE model, holding the best-validation-loss weights seen
        during training (not necessarily the final epoch's). Only present
        after calling `fit`.
    device_ : str
        The resolved torch device actually used for `model_`.
    """

    def __init__(
        self,
        motif_len=12,
        embed_size=10,
        hidden_size=32,
        kernel_size=7,
        fast=False,
        epochs=1000,
        batch_size=64,
        lr=1e-3,
        validation_fraction=0.1,
        threshold=50,
        beta_schedule=True,
        beta=1.0,
        force_matching=True,
        force_epochs=50,
        device=None,
        random_state=None,
    ):
        self.motif_len = motif_len
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.fast = fast
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.validation_fraction = validation_fraction
        self.threshold = threshold
        self.beta_schedule = beta_schedule
        self.beta = beta
        self.force_matching = force_matching
        self.force_epochs = force_epochs
        self.device = device
        self.random_state = random_state

    def _build_model(self):
        """Helper function to instantiate an untrained VAE
        matching this pipeline's params.
        """
        model_cls = CNN_PHMM_VAE_FAST if self.fast else CNN_PHMM_VAE
        return model_cls(
            motif_len=self.motif_len,
            embed_size=self.embed_size,
            hidden_size=self.hidden_size,
            kernel_size=self.kernel_size,
        )

    def _train_val_split(self, n):
        """Helper function to split input data into train and validation sets."""
        if n < 2:
            raise ValueError(f"Need at least 2 sequences to fit, got {n}.")
        n_val = max(1, int(n * self.validation_fraction))
        perm = torch.randperm(n)
        return perm[n_val:], perm[:n_val]

    def fit(self, X_idx, y=None):
        """Train the VAE on a collection of aptamer sequences."""
        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        self.device_ = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_ = self._build_model().to(self.device_)

        train_idx, val_idx = self._train_val_split(len(X_idx))
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_idx[train_idx]),
            batch_size=self.batch_size,
            shuffle=True,
        )
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_idx[val_idx]),
            batch_size=self.batch_size,
        )
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)

        beta = self.beta
        patient = 0
        best_val_loss = float("inf")
        best_state = None

        for epoch in range(1, self.epochs + 1):
            if self.beta_schedule and epoch < self.threshold:
                beta = epoch / self.threshold

            self.model_.train()
            train_loss = 0
            for (data,) in train_loader:
                data = data.to(self.device_)
                optimizer.zero_grad()
                if self.force_matching and epoch <= self.force_epochs:
                    loss = self.model_.loss_fn(
                        data,
                        *self.model_(data),
                        beta=beta,
                        force_matching=self.force_matching,
                        match_cost=1 + 4 * (1 - epoch / self.force_epochs),
                    )
                else:
                    loss = self.model_.loss_fn(data, *self.model_(data), beta=beta)
                loss.backward()
                train_loss += loss.item() * data.shape[0]
                optimizer.step()
            train_loss /= len(train_loader.dataset)
            if np.isnan(train_loss):
                break

            self.model_.eval()
            test_ce = 0
            test_kld = 0
            with torch.no_grad():
                for (data,) in val_loader:
                    data = data.to(self.device_)
                    ce, kld = self.model_.loss_fn(
                        data, *self.model_(data), beta=beta, test=True
                    )
                    test_ce += ce * data.shape[0]
                    test_kld += kld * data.shape[0]
            test_ce /= len(val_loader.dataset)
            test_kld /= len(val_loader.dataset)
            test_loss = test_ce + test_kld
            if np.isnan(test_loss):
                break

            if test_loss < best_val_loss:
                best_val_loss = test_loss
                best_state = {
                    k: v.detach().clone() for k, v in self.model_.state_dict().items()
                }
                patient = 0
            else:
                patient += 1
                if patient > self.threshold:
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        return self

    def transform(self, X):
        """Encode sequences into latent-space points.

        Parameters
        ----------
        X : Integer mapping of the input sequences, shape (n_samples, seq_len)
            Input sequences to encode.
        """
        check_is_fitted(self, "model_")
        self.model_.eval()

        X_idx = torch.as_tensor(X, dtype=torch.long).to(self.device_)
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
            To specify the sampling strategy.
        Returns
        -------
        sequences : list of str
            One fully-decoded sequence per latent-space point in `Z`.
        """
        check_is_fitted(self, "model_")
        self.model_.eval()

        z = torch.as_tensor(np.asarray(Z), dtype=torch.float32).to(self.device_)
        with torch.no_grad():
            transition_proba, emission_proba = self.model_.decoder(z)

        sequences = []
        for i in range(z.shape[0]):
            sampler = ProfileHMMSampler(
                transition_proba[i].cpu().numpy(),
                emission_proba[i].cpu().numpy(),
                proba_is_log=True,
            )
            if most_likely:
                seq = self._decode_most_probable(
                    sampler
                )  # most probable is called in decode_most_probable
            else:
                seq = sampler.sample(
                    sequence_only=True
                )  # sample alreadyreturn the full sequence
            sequences.append(seq)
        return sequences

    def _decode_most_probable(self, sampler, eval_max=256):
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

        if (
            len(all_combos) > eval_max
        ):  # if too many combinations, sample a subset of them
            idx = np.random.choice(len(all_combos), size=eval_max, replace=False)
            all_combos = [all_combos[i] for i in idx]

        candidates = []
        for combo in all_combos:
            combo_iter = iter(combo)
            candidates.append(
                "".join(next(combo_iter) if ch == "*" else ch for ch in pattern)
            )

        scored = [(c, sampler.calc_seq_proba(c)) for c in candidates]
        best_seq, _ = max(scored, key=lambda x: x[1])  # reutnr just the best candidate
        return best_seq
