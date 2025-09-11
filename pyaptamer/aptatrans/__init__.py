"""
AptaTrans pipeline and deep neural network for predicting aptamer-protein interaction
(API) and recommending candidate aptamers for a given target protein.
"""

__author__ = ["nennomp"]
__all__ = [
    "AptaTrans",
    "AptaTransPipeline",
    "EncoderPredictorConfig",
]

from pyaptamer.aptatrans._model import AptaTrans
from pyaptamer.aptatrans._pipeline import AptaTransPipeline
from pyaptamer.aptatrans.layers import EncoderPredictorConfig
