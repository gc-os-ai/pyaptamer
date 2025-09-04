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
    """Common base for supervised tabular predictors (classification/regression)."""

    def __init__(self, task="classification", random_state=None, verbose=0):
        self.task = task
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
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
