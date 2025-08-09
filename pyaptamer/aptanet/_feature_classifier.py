__author__ = "satvshr"
__all__ = ["AptaNetFeaturesClassifier"]
__required__ = ["python>=3.9,<3.12"]
import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted, validate_data
from torch import optim

from pyaptamer.aptanet._aptanet_nn import AptaNetMLP


class AptaNetFeaturesClassifier(ClassifierMixin, BaseEstimator):
    """
    Sklearn-compliant estimator that expects **numeric features** as X (2D array-like)
    and performs:
      (i) tree-based feature selection (RandomForest + SelectFromModel)
      (ii) skorch-wrapped AptaNetMLP binary classifier.

    This class should be covered by sklearn's estimator checks.
    """

    def __init__(
        self,
        input_dim=None,
        hidden_dim=128,
        n_hidden=7,
        dropout=0.3,
        max_epochs=20,
        lr=0.00014,
        alpha=0.9,
        eps=1e-08,
        n_estimators=300,
        max_depth=9,
        random_state=None,
        threshold="mean",
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden
        self.dropout = dropout
        self.max_epochs = max_epochs
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.threshold = threshold

    def _build_pipeline(self):
        from skorch import NeuralNetBinaryClassifier

        selector = SelectFromModel(
            estimator=RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
            ),
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
        )

        return Pipeline([("select", selector), ("net", net)])

    def fit(self, X, y):
        X, y = validate_data(self, X, y)
        # self.classes_, y = np.unique(y, return_inverse=True)
        # if np.unique(y).size < 2:
        #     raise ValueError("AptaNetFeaturesClassifier requires at least two classes
        # in y.")
        # 2) encode labels to {0,1} BEFORE casting to float
        # self.classes_, y = np.unique(y, return_inverse=True)
        self.pipeline_ = self._build_pipeline()
        X = X.astype(np.float32, copy=False)
        y = y.astype(np.float32, copy=False)
        self.pipeline_.fit(X, y)
        # self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        X = X.astype(np.float32, copy=False)

        # y01 = self.pipeline_.predict(X).astype(int, copy=False)
        # return self.classes_[y01]
        return self.pipeline_.predict(X)
