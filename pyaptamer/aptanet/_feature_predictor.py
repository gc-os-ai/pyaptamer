__author__ = ["nennomp", "satvshr"]
__all__ = ["AptaNetPredictor"]

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, r2_score
from sklearn.pipeline import Pipeline
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted, validate_data
from torch import optim

from pyaptamer.aptanet._aptanet_nn import AptaNetMLP
from pyaptamer.utils.tag_checks import task_check


class _BaseTabularSupervised(BaseEstimator):
    """
    Common base for AptaNet-style supervised tabular estimators supporting
    both classification and regression.

    This base class wires up a 2-stage pipeline: a tree-based
    `SelectFromModel` feature selector and a skorch-wrapped multi-layer
    perceptron (`AptaNetMLP`). Subclasses are responsible for constructing
    the exact pipeline in `_build_pipeline` (e.g., choosing classifier vs
    regressor variants and losses).

    Parameters
    ----------
    task : {"classification", "regression"}, default="classification"
        Whether the estimator behaves as a classifier or regressor.
    random_state : int or None, default=None
        Random seed for reproducibility. When set, both NumPy and Torch seeds are fixed.
    verbose : int, default=0
        Verbosity level propagated to the underlying skorch network.
    """

    def __init__(self, task="classification", random_state=None, verbose=0):
        self.task = task
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        """
        Fit the AptaNet pipeline to data.

        This method validates inputs, applies feature selection using
        a tree-based estimator, and trains the underlying skorch-wrapped
        neural network.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values. Must be binary if task="classification",
            continuous if task="regression".

        Returns
        -------
        self : object
            Fitted estimator.
        """
        task_check(self)
        X, y = validate_data(self, X, y)

        if self.task == "classification":
            y_type = type_of_target(y, raise_unknown=True)
            if y_type != "binary":
                raise ValueError(
                    "Only binary classification is supported. The type of the target "
                    f"is {y_type}."
                )
            self.classes_, y = np.unique(y, return_inverse=True)
        else:
            y = y.reshape(-1, 1)

        if self.random_state is not None:
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)

        self.pipeline_ = self._build_pipeline()
        X = X.astype(np.float32, copy=False)
        y = y.astype(np.float32, copy=False)
        self.pipeline_.fit(X, y)
        return self

    def predict(self, X):
        """
        Predict using the fitted AptaNet pipeline.

        For classification, this returns class labels. For regression,
        this returns continuous values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            Predicted class labels or regression values.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        X = X.astype(np.float32, copy=False)
        y = self.pipeline_.predict(X)

        if self.task == "regression":
            return y.reshape(-1)
        else:
            y = y.astype(int, copy=False)
            return self.classes_[y]

    def score(self, X, y):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        X = X.astype(np.float32, copy=False)
        y = y.astype(np.float32, copy=False)
        y_pred = self.predict(X)

        if self.task == "classification":
            return accuracy_score(y, y_pred)
        else:
            return r2_score(y, y_pred)


class AptaNetPredictor(_BaseTabularSupervised):
    """
    This estimator applies a tree-based `SelectFromModel` using a RandomForest
    to filter input features, then trains a skorch-wrapped multi-layer perceptron
    (`AptaNetMLP`). This mirrors the AptaNet-style deep model used for
    aptamer–protein interaction prediction [1]_.

    This estimator builds an internal sklearn `Pipeline` and delegates `fit`,
    `predict`, and other methods to it, while exposing convenient knobs for both
    the selector and the neural network. It supports both binary classification
    (BCE-with-logits) and regression (MSE), controlled via the `task` parameter.

    Parameters
    ----------
    task : {"classification", "regression"}, default="classification"
        Whether the estimator is a classifier (binary) or regressor.
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
        Estimator used for feature selection. If `None`, a `RandomForestClassifier`
        is used when `task="classification"` and a `RandomForestRegressor` when
        `task="regression"`.
    random_state : int or None, default=None
        Random seed for reproducibility. When set, both NumPy and Torch seeds are fixed.
    threshold : str or float, default="mean"
        Threshold passed to `SelectFromModel` (e.g., "mean" or a float).
    verbose : int, default=0
        Verbosity level for the underlying skorch network.

    References
    ----------

    .. [1] Emami, N., Ferdousi, R. AptaNet as a deep learning approach for
      aptamer–protein interaction prediction. Sci Rep 11, 6074 (2021).
      https://doi.org/10.1038/s41598-021-85629-0
    .. [2] https://github.com/nedaemami/AptaNet
    .. [3] https://www.nature.com/articles/s41598-021-85629-0.pdf

    Examples
    --------
    >>> from pyaptamer.aptanet import AptaNetPredictor
    >>> import numpy as np
    >>> X = np.random.rand(40, 128).astype(np.float32)
    >>> y = np.random.randint(0, 2, size=40)
    >>> clf = AptaNetPredictor(task="classification", input_dim=128, max_epochs=1)
    >>> clf.fit(X, y)
    >>> preds = clf.predict(X)
    """

    _tags = {"tasks": ["classification", "regression"]}

    def __init__(
        self,
        task="classification",
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
        super().__init__(task=task, random_state=random_state, verbose=verbose)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden
        self.dropout = dropout
        self.max_epochs = max_epochs
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.estimator = estimator
        self.threshold = threshold

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.non_deterministic = True

        if self.task == "classification":
            from sklearn.utils._tags import ClassifierTags

            tags.estimator_type = "classifier"
            tags.classifier_tags = ClassifierTags()
            tags.classifier_tags.poor_score = True
            tags.classifier_tags.multi_class = False
        else:
            from sklearn.utils._tags import RegressorTags

            tags.estimator_type = "regressor"
            tags.regressor_tags = RegressorTags()
            tags.regressor_tags.poor_score = True
        return tags

    def _build_pipeline(self):
        from skorch import NeuralNetBinaryClassifier, NeuralNetRegressor

        if self.task == "classification":
            base_estimator = self.estimator or RandomForestClassifier(
                n_estimators=300, max_depth=9, random_state=self.random_state
            )
        else:
            base_estimator = self.estimator or RandomForestRegressor(
                n_estimators=300, max_depth=9, random_state=self.random_state
            )

        selector = SelectFromModel(
            estimator=clone(base_estimator),
            threshold=self.threshold,
        )

        if self.task == "classification":
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
        else:
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

    def predict_proba(self, X):
        """
        Predict class probabilities for classification tasks.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            Predicted class probabilities.

        Raises
        ------
        AttributeError
            If called when task="regression".
        """
        if self.task != "classification":
            raise AttributeError(
                "predict_proba is only available when task='classification'."
            )
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        X = X.astype(np.float32, copy=False)
        return self.pipeline_.predict_proba(X)
