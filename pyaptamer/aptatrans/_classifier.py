"""AptaTrans classifier on aptamer and protein token ids."""

__author__ = ["siddharth7113"]
__all__ = ["AptaTransClassifier"]

import copy
import tempfile

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils.validation import check_is_fitted, validate_data
from torch.utils.data import DataLoader, TensorDataset

from pyaptamer.aptatrans._model import AptaTrans
from pyaptamer.aptatrans._model_lightning import AptaTransLightning
from pyaptamer.aptatrans.layers import EncoderPredictorConfig

_APTA_TARGET_DIM = 344
_PROT_TARGET_DIM = 585
_MIN_BLOCK_LEN = 6


class AptaTransClassifier(ClassifierMixin, BaseEstimator):
    """AptaTrans binary classifier for aptamer-protein interaction [1]_.

    Wraps the ``AptaTrans`` network in a scikit-learn classifier. Training
    runs through a Lightning ``Trainer`` with the steps defined in
    ``AptaTransLightning``. A fraction of ``X`` is held out to select the
    weights with the lowest validation loss and to stop early.

    Each row of ``X`` holds integer token ids. The first ``apta_max_len``
    columns are the aptamer tokens, as made by ``KMerIndexEncoder``. The other
    columns are the protein tokens, as made by ``GreedyEncoder``. Id 0 is
    padding. ``AptaTransPipeline`` builds this layout from sequences.

    The network needs at least 6 positions per block. A shorter block is
    padded with id 0.

    Parameters
    ----------
    apta_max_len : int, default=275
        Number of leading columns of ``X`` that hold aptamer token ids.
    n_apta_tokens : int, default=127
        Size of the aptamer vocabulary: 125 triplets, padding and mask.
    n_prot_tokens : int, default=715
        Size of the protein vocabulary: 713 protein words, padding and mask.
    in_dim : int, default=128
        Embedding size of the encoders.
    n_encoder_layers : int, default=6
        Number of transformer layers in each encoder.
    n_heads : int, default=8
        Number of attention heads. Must divide ``in_dim``.
    conv_layers : list of int or None, default=None
        Number of convolutional blocks in each of the 3 layers. If None,
        uses ``[3, 3, 3]``.
    dropout : float, default=0.1
        Dropout rate of the encoders.
    model : AptaTrans or None, default=None
        Starting network, for example one with pretrained encoders. ``fit``
        trains a copy of it instead of a new network built from the size
        parameters above and ``pretrained``.
    pretrained : bool, default=False
        If True, start from the pretrained AptaTrans weights. These need the
        default ``apta_max_len``, vocabulary sizes and network sizes, and a
        protein block of 867 columns.
    lr : float, default=1e-5
        Learning rate of the Adam optimizer.
    weight_decay : float, default=1e-5
        Weight decay of the Adam optimizer.
    max_epochs : int, default=200
        Maximum number of training epochs.
    patience : int or None, default=10
        Stop training after this many epochs without improvement of the
        validation loss. None disables early stopping.
    batch_size : int, default=16
        Minibatch size for training and prediction.
    validation_fraction : float, default=0.05
        Fraction of ``X`` held out in each ``fit`` call. At least one row is
        held out.
    random_state : int or None, default=None
        Seed for the weight initialization, the validation split and the
        training order.
    verbose : int, default=0
        If non-zero, show the Lightning progress bar and model summary.

    Attributes
    ----------
    classes_ : ndarray of shape (2,)
        Class labels seen in ``fit``.
    model_ : AptaTrans
        The fitted network, with the weights of the epoch with the lowest
        validation loss.
    n_epochs_ : int
        Number of epochs run during ``fit``.

    References
    ----------
    .. [1] Shin, Incheol, et al. "AptaTrans: a deep neural network for predicting
       aptamer-protein interaction using pretrained encoders." BMC bioinformatics
       24.1 (2023): 447.

    Examples
    --------
    >>> import numpy as np
    >>> from pyaptamer.aptatrans import AptaTransClassifier
    >>> rng = np.random.default_rng(0)
    >>> X = rng.integers(1, 20, size=(8, 16))
    >>> y = np.array([0, 1] * 4)
    >>> clf = AptaTransClassifier(
    ...     apta_max_len=8,
    ...     in_dim=8,
    ...     n_heads=2,
    ...     n_encoder_layers=1,
    ...     conv_layers=[1, 1, 1],
    ...     max_epochs=1,
    ...     random_state=0,
    ... )
    >>> clf.fit(X, y).predict_proba(X).shape
    (8, 2)
    """

    def __init__(
        self,
        apta_max_len=275,
        n_apta_tokens=127,
        n_prot_tokens=715,
        in_dim=128,
        n_encoder_layers=6,
        n_heads=8,
        conv_layers=None,
        dropout=0.1,
        model=None,
        pretrained=False,
        lr=1e-5,
        weight_decay=1e-5,
        max_epochs=200,
        patience=10,
        batch_size=16,
        validation_fraction=0.05,
        random_state=None,
        verbose=0,
    ):
        self.apta_max_len = apta_max_len
        self.n_apta_tokens = n_apta_tokens
        self.n_prot_tokens = n_prot_tokens
        self.in_dim = in_dim
        self.n_encoder_layers = n_encoder_layers
        self.n_heads = n_heads
        self.conv_layers = conv_layers
        self.dropout = dropout
        self.model = model
        self.pretrained = pretrained
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        """Train the network on token-id rows and binary labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Aptamer token ids in the first ``apta_max_len`` columns, protein
            token ids in the others.
        y : array-like of shape (n_samples,)
            Binary class labels.

        Returns
        -------
        self : AptaTransClassifier
        """
        X, y = validate_data(self, X, y)
        check_classification_targets(y)
        y_type = type_of_target(y, input_name="y", raise_unknown=True)
        if y_type != "binary":
            raise ValueError(
                "Only binary classification is supported. The type of the target "
                f"is {y_type}."
            )
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        if len(self.classes_) != 2:
            raise ValueError(
                f"{type(self).__name__} is a binary classifier, but y has "
                f"{len(self.classes_)} class(es): {self.classes_.tolist()}."
            )
        x_apta, x_prot = self._to_tensors(X)
        y_t = torch.as_tensor(y_idx, dtype=torch.float32)
        with torch.random.fork_rng(enabled=self.random_state is not None):
            if self.random_state is not None:
                torch.manual_seed(self.random_state)
            self._fit(x_apta, x_prot, y_t)
        return self

    def _fit(self, x_apta, x_prot, y):
        n_val = max(1, int(len(y) * self.validation_fraction))
        perm = torch.randperm(len(y))
        train_idx, val_idx = perm[n_val:], perm[:n_val]
        train_loader = DataLoader(
            TensorDataset(x_apta[train_idx], x_prot[train_idx], y[train_idx]),
            batch_size=self.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(x_apta[val_idx], x_prot[val_idx], y[val_idx]),
            batch_size=self.batch_size,
        )
        model = AptaTransLightning(
            self._build_model(x_apta.shape[1], x_prot.shape[1]),
            lr=self.lr,
            weight_decay=self.weight_decay,
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

    def _build_model(self, apta_len, prot_len):
        """Copy the given network, or build one for blocks of the given lengths."""
        if self.model is not None:
            return copy.deepcopy(self.model)
        return AptaTrans(
            apta_embedding=EncoderPredictorConfig(
                num_embeddings=self.n_apta_tokens,
                target_dim=_APTA_TARGET_DIM,
                max_len=apta_len,
            ),
            prot_embedding=EncoderPredictorConfig(
                num_embeddings=self.n_prot_tokens,
                target_dim=_PROT_TARGET_DIM,
                max_len=prot_len,
            ),
            in_dim=self.in_dim,
            n_encoder_layers=self.n_encoder_layers,
            n_heads=self.n_heads,
            conv_layers=self.conv_layers,
            dropout=self.dropout,
            pretrained=self.pretrained,
        )

    def _to_tensors(self, X):
        """Split X into the aptamer and protein token tensors."""
        name = type(self).__name__
        if X.shape[1] <= self.apta_max_len:
            raise ValueError(
                f"X has n_features={X.shape[1]}, but apta_max_len="
                f"{self.apta_max_len} leaves no column for the protein tokens. Pass "
                f"{name}(apta_max_len=...) with a value below {X.shape[1]}."
            )
        if np.any(X < 0):
            raise ValueError(
                f"Negative values in data passed to {name}. X must hold token ids."
            )
        if np.any(X != np.round(X)):
            raise ValueError(f"{name} expects integer token ids in X.")

        x_apta = self._block(X[:, : self.apta_max_len], "aptamer", "n_apta_tokens")
        x_prot = self._block(X[:, self.apta_max_len :], "protein", "n_prot_tokens")
        return x_apta, x_prot

    def _block(self, block, block_name, param):
        """Check the ids of one block against the parameter named by param and pad
        the block to at least 6 columns."""
        n_tokens = getattr(self, param)
        if block.max() >= n_tokens:
            raise ValueError(
                f"The {block_name} columns of X hold token id {int(block.max())}, "
                f"but {param}={n_tokens}. Pass {type(self).__name__}({param}=...) "
                f"with a value above {int(block.max())}."
            )
        width = max(block.shape[1], _MIN_BLOCK_LEN)
        padded = np.zeros((len(block), width), dtype=np.int64)
        padded[:, : block.shape[1]] = block
        return torch.as_tensor(padded)

    def _forward(self, X, imap=False):
        """Run the network over X in minibatches and return a numpy array."""
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        x_apta, x_prot = self._to_tensors(X)
        self.model_.eval()
        forward = self.model_.forward_imap if imap else self.model_
        device = next(self.model_.parameters()).device
        loader = DataLoader(TensorDataset(x_apta, x_prot), batch_size=self.batch_size)
        with torch.no_grad():
            out = [
                forward(apta.to(device), prot.to(device)).cpu() for apta, prot in loader
            ]
        return torch.cat(out).numpy()

    def predict_proba(self, X):
        """Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Token-id rows in the layout used in ``fit``.

        Returns
        -------
        ndarray of shape (n_samples, 2)
            Probability of ``classes_[0]`` and ``classes_[1]``.
        """
        p_positive = self._forward(X)[:, 0].astype(np.float64)
        return np.column_stack([1 - p_positive, p_positive])

    def predict(self, X):
        """Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Token-id rows in the layout used in ``fit``.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X)
        return self.classes_[proba.argmax(axis=1)]

    def interaction_map(self, X):
        """Compute the aptamer-protein interaction map of each row.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Token-id rows in the layout used in ``fit``.

        Returns
        -------
        ndarray of shape (n_samples, 1, apta_len, prot_len)
            Interaction map between the aptamer and protein positions. Each
            length is the width of its block, at least 6.
        """
        return self._forward(X, imap=True)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.categorical = True
        tags.input_tags.positive_only = True
        tags.classifier_tags.multi_class = False
        tags.classifier_tags.poor_score = True
        return tags
