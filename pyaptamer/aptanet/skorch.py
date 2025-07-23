import torch.nn as nn
from skorch import NeuralNetBinaryClassifier

from pyaptamer.aptanet.aptanet_nn import AptaNetMLP


class SkorchAptaNet(NeuralNetBinaryClassifier):
    def __init__(self, threshold=0.5, **kwargs):
        self.threshold = threshold
        super().__init__(module=AptaNetMLP, criterion=nn.BCEWithLogitsLoss, **kwargs)

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba > self.threshold).astype(int)
