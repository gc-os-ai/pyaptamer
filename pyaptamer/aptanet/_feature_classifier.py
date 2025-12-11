__author__ = ["nennomp", "satvshr"]
__all__ = ["AptaNetClassifier", "AptaNetRegressor"]

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted, validate_data
from torch import optim

from pyaptamer.aptanet._aptanet_nn import AptaNetMLP


class AptaNetClassifier(ClassifierMixin, BaseEstimator):
    """
    AptaNet-style binary classifier for aptamer–protein interaction prediction.

    This estimator applies a tree-based `SelectFromModel` (RandomForest)
    to filter input features, then trains a skorch-wrapped multi-layer
    perceptron (`AptaNetMLP`) with BCE-with-logits loss. [1]_

    This classifier builds an internal sklearn `Pipeline` and delegates `fit`,
    `predict`, and other methods to it, while exposing convenient knobs for both
    the selector and the neural network.

    The estimator is non-deterministic and only supports binary classification.

    References
    ----------
    .. [1] Emami, N., Ferdousi, R. AptaNet as a deep learning approach for
       aptamer–protein interaction prediction. Sci Rep 11, 6074 (2021).
       https://doi.org/10.1038/s41598-021-85629-0
    .. [2] https://github.com/nedaemami/AptaNet
    .. [3] https://www.nature.com/articles/s41598-021-85629-0.pdf
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
            estimator=clone(base_estimator), threshold=self.threshold
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
        """
        Fit the classifier on training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Binary class labels (0/1).

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

        if self.random_state is not None:
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)

        self.classes_, y = np.unique(y, return_inverse=True)
        self.pipeline_ = self._build_pipeline()
        self.pipeline_.fit(
            X.astype(np.float32, copy=False), y.astype(np.float32, copy=False)
        )
        self.is_fitted_ = True
        return self

    def predict_proba(self, X):
        """
        Predict class probabilities for samples in `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Probability estimates for each class.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False).astype(np.float32, copy=False)
        return self.pipeline_.predict_proba(X)

    def predict(self, X):
        """
        Predict binary class labels for samples in `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False).astype(np.float32, copy=False)
        y = self.pipeline_.predict(X).astype(int, copy=False)
        return self.classes_[y]

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        tags.classifier_tags.poor_score = True
        tags.non_deterministic = True
        tags.requires_fit = True
        return tags


class AptaNetRegressor(RegressorMixin, BaseEstimator):
    """
    AptaNet-style regressor for aptamer–protein interaction prediction.

    This estimator applies a tree-based `SelectFromModel` (RandomForest)
    to filter input features, then trains a skorch-wrapped multi-layer
    perceptron (`AptaNetMLP`) with mean squared error loss. [1]_

    This regressor builds an internal sklearn `Pipeline` and delegates `fit`,
    `predict`, and other methods to it, while exposing convenient knobs for both
    the selector and the neural network.

    The estimator is non-deterministic and supports continuous regression targets.

    References
    ----------
    .. [1] Emami, N., Ferdousi, R. AptaNet as a deep learning approach for
       aptamer–protein interaction prediction. Sci Rep 11, 6074 (2021).
       https://doi.org/10.1038/s41598-021-85629-0
    .. [2] https://github.com/nedaemami/AptaNet
    .. [3] https://www.nature.com/articles/s41598-021-85629-0.pdf

    Parameters
    ----------
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
        from skorch import NeuralNetRegressor

        base_estimator = self.estimator or RandomForestRegressor(
            n_estimators=300, max_depth=9, random_state=self.random_state
        )
        selector = SelectFromModel(
            estimator=clone(base_estimator), threshold=self.threshold
        )

        net = NeuralNetRegressor(
            module=AptaNetMLP,
            module__input_dim=self.input_dim,
            module__hidden_dim=self.hidden_dim,
            module__n_hidden=self.n_hidden,
            module__dropout=self.dropout,
            module__output_dim=1,
            module__use_lazy=True,
            criterion=nn.MSELoss,
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
        """
        Fit the regressor on training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Continuous regression targets.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X, y = validate_data(self, X, y)
        y = y.reshape(-1, 1)

        if self.random_state is not None:
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)

        self.pipeline_ = self._build_pipeline()
        self.pipeline_.fit(
            X.astype(np.float32, copy=False), y.astype(np.float32, copy=False)
        )

        return self

    def predict(self, X):
        """
        Predict continuous values for samples in `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted continuous values.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False).astype(np.float32, copy=False)

        return self.pipeline_.predict(X).reshape(-1)

    def score(self, X, y):
        from sklearn.metrics import r2_score

        return r2_score(y, self.predict(X))

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.regressor_tags.poor_score = True
        tags.requires_fit = True
        return tags
