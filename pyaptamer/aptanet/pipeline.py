__author__ = "satvshr"
__all__ = ["AptaPipeline"]
__required__ = ["python>=3.9,<3.12"]

import numpy as np
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data
from torch import optim

from pyaptamer.aptanet.aptanet_nn import AptaNetMLP
from pyaptamer.utils._aptanet_utils import pairs_to_features


class AptaPipeline(ClassifierMixin, BaseEstimator):
    """
    Pipeline for aptamer-pair classification combining feature extraction,
    tree-based feature selection, and a skorch-wrapped neural network.

    This class wraps an internal sklearn Pipeline and delegates fit, predict,
    and other methods to it, while allowing flexible configuration of each step.

    References
    ----------


    - Emami, N., Ferdousi, R. AptaNet as a deep learning approach for aptamer–protein
    interaction prediction. Sci Rep 11, 6074 (2021). https://doi.org/10.1038/s41598-021-85629-0
    - https://github.com/nedaemami/AptaNet
    - https://www.nature.com/articles/s41598-021-85629-0.pdf


    Parameters
    ----------
    input_dim : int, default=128
        Size of the input layer in the neural net.
    hidden_dim : int, default=64
        Number of units in each hidden layer of the neural net.
    n_hidden : int, default=5
        Number of hidden layers in the neural net.
    dropout : float, default=0.3
        Dropout probability in the neural net.
    max_epochs : int, default=20
        Maximum number of training epochs for the neural net.
    lr : float, default=0.00014
        Learning rate for the optimizer (RMSprop).
    alpha : float, default=0.9
        Discounting factor (rho) for the squared‐gradient moving average in RMSprop.
    eps : float or None, default=None
        Epsilon value for numerical stability in RMSprop; if None, PyTorch’s default
        (1e-08) is used.
    n_estimators : int, default=300
        Number of trees in the RandomForest feature selector.
    max_depth : int, default=9
        Maximum depth of each tree in the RandomForest.
    random_state : int, default=0
        Random seed for reproducibility.
    threshold : str or float, default="mean"
        Threshold for SelectFromModel (e.g. "mean" or a float).
    pairs_to_features_kwargs : dict, default=None
        Extra keyword arguments passed directly to `pairs_to_features`. Valid keys are:


        - k : int, optional, default=4
          The k-mer size for generating aptamer k-mer vectors.
        - pseaac_kwargs : dict, optional, default=None


        Example:
            pairs_to_features_kwargs = {
                "k": 4,
                "pseaac_kwargs": {
                    "lambda_value": 30,
                    "w": 0.05
                }
            }

    Examples
    --------
    >>> from pyaptamer.aptanet.pipeline import AptaPipeline
    >>> pipe = AptaPipeline(max_epochs=50, lr=0.005)
    >>> pipe.fit(X_train, y_train)
    >>> preds = pipe.predict(X_test)
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
        pairs_to_features_kwargs=None,
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
        self.pairs_to_features_kwargs = pairs_to_features_kwargs or {}
        self.pipeline_ = None
        self.classes_ = None

    # ---------------------- internal helpers ----------------------

    def _build_pipeline(self):
        """Build the inner sklearn Pipeline. Lazy-import skorch here."""
        try:
            from skorch import NeuralNetBinaryClassifier  # lazy import
        except Exception as e:
            raise ImportError(
                "AptaPipeline requires 'skorch' at runtime. "
                "Please install skorch (and a supported Python/PyTorch) to use the NN"
                "step."
            ) from e

        features = FunctionTransformer(
            func=pairs_to_features,
            validate=False,  # let raw strings pass through; conversion happens inside
            kw_args=self.pairs_to_features_kwargs,
        )

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
        )

        return Pipeline([("features", features), ("select", selector), ("net", net)])

    # ---------------------- sklearn API ----------------------

    def fit(self, X, y):
        # Disallow NaNs; allow non-numeric raw inputs to pass to transformer
        X, y = validate_data(
            self,
            X=X,
            y=y,
            reset=True,
            dtype=None,  # keep strings; conversion is downstream
            force_all_finite=True,  # <- No NaNs allowed
        )
        check_classification_targets(y)
        # encode labels to ints; remember mapping
        self.classes_, y_enc = np.unique(y, return_inverse=True)

        self.pipeline_ = self._build_pipeline()
        self.pipeline_.fit(X, y_enc)
        return self

    def predict(self, X):
        check_is_fitted(self, "pipeline_")
        X = validate_data(
            self,
            X=X,
            y="no_validation",  # do not require y at predict time
            reset=False,
            dtype=None,
            force_all_finite=True,  # <- No NaNs allowed
        )
        y_enc = self.pipeline_.predict(X)
        return self.classes_[y_enc]

    # ---------------------- sklearn test hints ----------------------

    def _more_tags(self):
        # Relax checks that assume numeric X pre-transform; note non-determinism.
        return {
            "X_types": ["string", "categorical", "2darray"],
            "non_deterministic": True,
            "requires_y": True,
            # intentionally not setting 'allow_nan' (NaNs are disallowed)
        }
