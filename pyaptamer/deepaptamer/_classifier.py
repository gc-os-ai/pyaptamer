__author__ = ["satvshr", "geetu040"]
__all__ = ["DeepAptamerClassifier"]

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted, validate_data
from torch import optim

from pyaptamer.deepaptamer._deepaptamer_nn import DeepAptamerNN


class DeepAptamerClassifier(ClassifierMixin, BaseEstimator):
    """
    DeepAptamer binary classifier for aptamer–protein interaction prediction.

    This estimator trains a skorch-wrapped `DeepAptamerNN` with cross-entropy loss
    and the Adam optimizer. [1]_

    This follows the original implementation [2]_.

    The network has two input branches, a one-hot encoded sequence and a DNA shape
    vector. Both are passed in a single feature matrix `X`: the first
    ``4 * seq_len`` columns hold the one-hot sequence, flattened row-wise from shape
    (`seq_len`, 4), and the remaining columns hold the shape vector.

    The estimator is non-deterministic and only supports binary classification.

    Parameters
    ----------
    seq_len : int, default=35
        Length of the (padded) aptamer sequences. Used to split `X` into the
        one-hot and shape parts.
    dropout : float, default=0.1
        Dropout probability used in the neural net.
    max_epochs : int, default=100
        Maximum number of training epochs for the neural net.
    batch_size : int, default=256
        Mini-batch size for training and inference.
    lr : float, default=0.001
        Learning rate for the optimizer (Adam).
    random_state : int or None, default=None
        Seed for Torch during ``fit``. The global Torch random state is left
        unchanged.
    verbose : int, default=0
        Verbosity level for the underlying skorch `NeuralNetClassifier`.

    Attributes
    ----------
    classes_ : ndarray of shape (2,)
        Class labels seen during ``fit``.
    net_ : skorch.NeuralNetClassifier
        The fitted network.

    References
    ----------
    .. [1] Yang X, Chan CH, Yao S, Chu HY, Lyu M, Chen Z, Xiao H, Ma Y, Yu S, Li F,
       Liu J, Wang L, Zhang Z, Zhang BT, Zhang L, Lu A, Wang Y, Zhang G, Yu Y.
       DeepAptamer: Advancing high-affinity aptamer discovery with a hybrid deep
       learning model. Mol Ther Nucleic Acids. 2024 Dec 21;36(1):102436.
       doi: 10.1016/j.omtn.2024.102436.
    .. [2] DeepAptamer original implementation.
       https://github.com/YangX-BIDD/DeepAptamer
    """

    def __init__(
        self,
        seq_len=35,
        dropout=0.1,
        max_epochs=100,
        batch_size=256,
        lr=0.001,
        random_state=None,
        verbose=0,
    ):
        self.seq_len = seq_len
        self.dropout = dropout
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.random_state = random_state
        self.verbose = verbose

    def _build_net(self):
        from skorch import NeuralNetClassifier

        return NeuralNetClassifier(
            module=DeepAptamerNN,
            module__dropout=self.dropout,
            criterion=nn.CrossEntropyLoss,
            optimizer=optim.Adam,
            lr=self.lr,
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            device="cuda" if torch.cuda.is_available() else "cpu",
            verbose=self.verbose,
        )

    def _split(self, X):
        """Split the flat feature matrix into the two network inputs."""
        n_ohe = 4 * self.seq_len
        if X.shape[1] <= n_ohe:
            raise ValueError(
                f"X has {X.shape[1]} features, but seq_len={self.seq_len} needs "
                f"{n_ohe} one-hot features followed by the shape vector."
            )
        X = X.astype(np.float32, copy=False)
        return {
            "x_ohe": X[:, :n_ohe].reshape(-1, self.seq_len, 4),
            "x_shape": X[:, n_ohe:].reshape(len(X), 1, -1),
        }

    def fit(self, X, y):
        """
        Fit the classifier on training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, 4 * seq_len + shape_len)
            Training features, one-hot sequence followed by the shape vector.
        y : array-like of shape (n_samples,)
            Binary class labels.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X, y = validate_data(self, X, y)
        y_type = type_of_target(y, input_name="y", raise_unknown=True)
        if y_type != "binary":
            raise ValueError(
                f"Only binary classification is supported. Got target type {y_type}."
            )

        self.classes_, y = np.unique(y, return_inverse=True)
        with torch.random.fork_rng(enabled=self.random_state is not None):
            if self.random_state is not None:
                torch.manual_seed(self.random_state)
            self.net_ = self._build_net()
            self.net_.fit(self._split(X), y.astype(np.int64, copy=False))
        return self

    def predict_proba(self, X):
        """
        Predict class probabilities for samples in `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, 4 * seq_len + shape_len)
            Input features, one-hot sequence followed by the shape vector.

        Returns
        -------
        ndarray of shape (n_samples, 2)
            Probability estimates for each class, in the order of `classes_`.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        return self.net_.predict_proba(self._split(X))

    def predict(self, X):
        """
        Predict binary class labels for samples in `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, 4 * seq_len + shape_len)
            Input features, one-hot sequence followed by the shape vector.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        tags.classifier_tags.poor_score = True
        tags.non_deterministic = True
        return tags
