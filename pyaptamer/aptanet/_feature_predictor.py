import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted, validate_data
from torch import optim

from pyaptamer.aptanet._aptanet_nn import AptaNetMLP
from pyaptamer.utils.tag_check import task_check


class AptaNetPredictor(BaseEstimator):
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
        self.task = task
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

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()

        # mark type
        if self.task == "classification":
            tags.estimator_type = "classifier"
            try:
                from sklearn.utils._tags import ClassifierTags

                tags.classifier_tags = ClassifierTags()
                tags.classifier_tags.poor_score = True
            except ImportError:
                pass
        else:
            tags.estimator_type = "regressor"
            try:
                from sklearn.utils._tags import RegressorTags

                tags.regressor_tags = RegressorTags()
            except ImportError:
                pass

        return tags

    def _build_pipeline(self):
        from skorch import NeuralNetBinaryClassifier, NeuralNetRegressor

        base_estimator = self.estimator or RandomForestClassifier(
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

    def fit(self, X, y):
        task_check(self)
        X, y = validate_data(self, X, y)

        if self.task == "classification" and "continuous" in type_of_target(y):
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
        y = self.pipeline_.predict(X)
        if self.task == "regression":
            y = y.reshape(-1)

        else:  # classification
            y = y.astype(int, copy=False)
            y = self.classes_[y]
        # y = self.pipeline_.predict(X).astype(int, copy=False)
        # y = self.pipeline_.predict(X)
        # return self.classes_[y]
        return y
