"""
AptaTrans pipeline and deep neural network for predicting aptamer-protein interaction
(API) and generating candidate aptamers for a given target protein.
"""

__author__ = ["nennomp"]
__all__ = [
    "AptaTrans",
    "AptaTransClassifier",
    "AptaTransLightning",
    "AptaTransEncoderLightning",
    "AptaTransPipeline",
    "EncoderPredictorConfig",
]

from pyaptamer.aptatrans._classifier import AptaTransClassifier
from pyaptamer.aptatrans._model import AptaTrans
from pyaptamer.aptatrans._model_lightning import (
    AptaTransEncoderLightning,
    AptaTransLightning,
)
from pyaptamer.aptatrans._pipeline import AptaTransPipeline
from pyaptamer.aptatrans.layers import EncoderPredictorConfig
