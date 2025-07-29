__author__ = "satvshr"
__all__ = ["SkorchAptaNet"]

import torch.nn as nn
from skorch import NeuralNetBinaryClassifier

from pyaptamer.aptanet.aptanet_nn import AptaNetMLP


class SkorchAptaNet(NeuralNetBinaryClassifier):
    """
    A Skorch-based binary classifier using AptaNetMLP with a configurable threshold.

    This class wraps a PyTorch-based multilayer perceptron (MLP) model `AptaNetMLP`
    using the Skorch API. It is designed for binary classification tasks, using the
    binary cross-entropy loss with logits (`BCEWithLogitsLoss`) and a configurable
    threshold for converting predicted probabilities to class labels.

    Parameters
    ----------
    threshold : float, optional (default=0.5)
        The decision threshold for converting probabilities into binary predictions.
        Probabilities above this threshold are mapped to class 1, otherwise to class 0.

    **kwargs : dict
        Additional keyword arguments passed to `NeuralNetBinaryClassifier`.
        To configure the underlying `AptaNetMLP` module, use the `module__<param>`
        syntax. For example, `module__hidden_size=128` will pass `hidden_size=128`
        to `AptaNetMLP`.

    Methods
    -------
    predict(X)
        Predicts class labels for the input data `X` using the fitted model.
        Returns binary predictions based on the specified threshold.
    """

    def __init__(self, threshold=0.5, **kwargs):
        self.threshold = threshold
        super().__init__(module=AptaNetMLP, criterion=nn.BCEWithLogitsLoss, **kwargs)

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba > self.threshold).astype(int)
