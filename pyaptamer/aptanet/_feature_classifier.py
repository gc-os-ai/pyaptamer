__author__ = "satvshr"
__all__ = ["AptaNetClassifier"]

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted, validate_data
from torch import optim

from pyaptamer.aptanet._model import AptaNetMLP


class AptaNetClassifier(ClassifierMixin, BaseEstimator):
    """
    This estimator applies a tree-based `SelectFromModel` using a RandomForest
    to filter input features, then trains a skorch-wrapped multi-layer perceptron
    (`AptaNetMLP`) with BCE-with-logits. This mirrors the AptaNet-style deep
    model used for aptamer–protein interaction prediction.

    This classifier builds an internal sklearn `Pipeline` and delegates `fit`,
    `predict`, and other methods to it, while exposing convenient knobs for both
    the selector and the neural network.

    References
    ----------


    - Emami, N., Ferdousi, R. AptaNet as a deep learning approach for
    aptamer–protein interaction prediction. Sci Rep 11, 6074 (2021).
    https://doi.org/10.1038/s41598-021-85629-0
    - https://github.com/nedaemami/AptaNet
    - https://www.nature.com/articles/s41598-021-85629-0.pdf


    Parameters
    ----------
    input_dim : int or None, default=None
        Size of the input layer in the neural net. If `None`, it should be
        inferred from the feature matrix shape by the underlying module.
    hidden_dim : int, default=128
        Number of units in each hidden layer of the neural net.
    n_hidden : int, default=7
        Number of hidden layers in the neural net.
    dropout : float, default=0.3
        Dropout probability used in the neural net.
    max_epochs : int, default=200
        Maximum number of training epochs for the neural net.
    lr : float, default=0.00014
        Learning rate for the optimizer (RMSprop).
    alpha : float, default=0.9
        Discounting factor (rho) for the squared-gradient moving average in RMSprop.
    eps : float, default=1e-08
        Epsilon value for numerical stability in RMSprop.
    estimator : sklearn estimator or None, default=None
        Estimator used for feature selection. If `None`, a `RandomForestClassifier`.
    random_state : int or None, default=None
        Random seed for reproducibility. When set, both NumPy and Torch seeds are fixed.
    threshold : str or float, default="mean"
        Threshold passed to `SelectFromModel` (e.g., "mean" or a float).
    verbose : int, default=0
        Verbosity level for the underlying skorch `NeuralNetBinaryClassifier`.
    """

    def __init__(
        self,
        input_dim=None,
        hidden_dim=128,
        n_hidden=7,
        dropout=0.3,
        max_epochs=200,
        lr=0.00014,
        alpha=0.9,
        eps=1e-08,
        estimator=None,
        random_state=None,
        threshold="mean",
        verbose=0,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden
        self.dropout = dropout
        self.max_epochs = max_epochs
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.estimator = estimator
        self.random_state = random_state
        self.threshold = threshold
        self.verbose = verbose

    def _build_pipeline(self):
        from skorch import NeuralNetBinaryClassifier

        base_estimator = self.estimator or RandomForestClassifier(
            n_estimators=300, max_depth=9, random_state=self.random_state
        )

        selector = SelectFromModel(
            estimator=clone(base_estimator),
            threshold=self.threshold,
        )

        net = NeuralNetBinaryClassifier(
            module=AptaNetMLP,
            module__input_dim=self.input_dim,
            module__hidden_dim=self.hidden_dim,
            module__n_hidden=self.n_hidden,
            module__dropout=self.dropout,
            module__output_dim=1,
            module__use_lazy=True,
            criterion=nn.BCEWithLogitsLoss,
            max_epochs=self.max_epochs,
            lr=self.lr,
            optimizer=optim.RMSprop,
            optimizer__alpha=self.alpha,
            optimizer__eps=self.eps,
            device="cuda" if torch.cuda.is_available() else "cpu",
            verbose=self.verbose,
        )

        return Pipeline([("select", selector), ("net", net)])

    def fit(self, X, y):
        X, y = validate_data(self, X, y)

        if "continuous" in type_of_target(y):
            raise ValueError("continuous target is not supported for classification")

        if self.random_state is not None:
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)

        self.classes_, y = np.unique(y, return_inverse=True)

        self.pipeline_ = self._build_pipeline()
        X = X.astype(np.float32, copy=False)
        y = y.astype(np.float32, copy=False)
        self.pipeline_.fit(X, y)
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        X = X.astype(np.float32, copy=False)
        y = self.pipeline_.predict(X).astype(int, copy=False)
        return self.classes_[y]

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.poor_score = True
        return tags
