__author__ = "satvshr"
__all__ = ["SkorchAptaNet"]
__required__ = ["python>=3.9,<3.12"]

import torch.nn as nn
from skorch import NeuralNetBinaryClassifier

from pyaptamer.aptanet.aptanet_nn import AptaNetMLP


class SkorchAptaNet(NeuralNetBinaryClassifier):
    """
    A Skorch-based binary classifier wrapping AptaNetMLP, with configurable
    architecture and decision threshold for binary classification.

    This model uses BCEWithLogitsLoss and allows users to customize the MLP
    architecture using AptaNetMLP parameters, while maintaining compatibility
    with scikit-learn utilities like GridSearchCV and Pipelines.

    Parameters
    ----------
    threshold : float, default=0.5
        Threshold for converting predicted probabilities into class labels.

    module__input_dim : int or None, optional
        Input dimensionality. If None and `module__use_lazy=True`, lazy initialization
        is used.

    module__hidden_dim : int, default=128
        Number of hidden units per hidden layer.

    module__n_hidden : int, default=7
        Number of hidden layers.

    module__dropout : float, default=0.3
        Dropout probability for AlphaDropout in each hidden layer.

    module__output_dim : int, default=1
        Output dimensionality (typically 1 for binary classification).

    module__use_lazy : bool, default=True
        Whether to use `nn.LazyLinear` in the first layer.

    **kwargs : dict
        Additional parameters passed to `NeuralNetBinaryClassifier`, such as
        training parameters (e.g. `max_epochs`, `lr`, `optimizer`) or callbacks.
    """

    def __init__(
        self,
        threshold=0.5,
        module__input_dim=None,
        module__hidden_dim=128,
        module__n_hidden=7,
        module__dropout=0.3,
        module__output_dim=1,
        module__use_lazy=True,
        **kwargs,
    ):
        self.threshold = threshold
        super().__init__(
            module=AptaNetMLP,
            criterion=nn.BCEWithLogitsLoss,
            module__input_dim=module__input_dim,
            module__hidden_dim=module__hidden_dim,
            module__n_hidden=module__n_hidden,
            module__dropout=module__dropout,
            module__output_dim=module__output_dim,
            module__use_lazy=module__use_lazy,
            **kwargs,
        )

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba > self.threshold).astype(int)
